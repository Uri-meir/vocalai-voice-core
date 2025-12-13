from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import json
import base64
import asyncio
from src.telephony.audio_utils import mulaw_to_pcm, pcm_to_mulaw, resample_audio
from src.gemini.client import GeminiLiveClient
from src.config.environment import config

router = APIRouter()
logger = logging.getLogger(__name__)

from src.core.session import CallSession
from src.core.assistants_repository_factory import get_assistant_repository
from src.core.events.emitter import get_supabase_vapi_webhook_emitter
from datetime import datetime, timezone

@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio Media Stream WebSocket and Bridge to Gemini."""
    await websocket.accept()
    logger.info("🔌 Twilio Media Stream Connected")
    
    # queues for bridging
    mic_queue = asyncio.Queue()     # Twilio -> Gemini
    speaker_queue = asyncio.Queue() # Gemini -> Twilio
    
    stream_sid = None
    sender_task = None
    call_session = None
    
    # Call State
    call_start_time = None
    call_id = None
    assistant_id_webhook = None # ID to send to webhook
    customer_number = None

    try:
        # Inbound Loop (Twilio -> Gemini)
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")
            
            if event == "connected":
                logger.info(f"✅ Twilio Connected: {data}")
                
            elif event == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                call_id = data.get("start", {}).get("callSid")
                custom_params = data.get("start", {}).get("customParameters", {})
                
                # Internal ID used for config lookup
                internal_assistant_id = custom_params.get("assistant_id")
                
                # Attempt to get customer number from customParams (if passed from voice_hook)
                # Default to "unknown" if not available (e.g. if passed directly to TwiML without params)
                
                customer_number = custom_params.get("customer_number", "unknown") 

                logger.info(f"🏁 Stream Started: {stream_sid}, Assistant: {internal_assistant_id}")
                
                call_start_time = datetime.now(timezone.utc)

                # Load Configuration
                repo = get_assistant_repository()
                assistant_config = await repo.get_by_id(internal_assistant_id) if internal_assistant_id else None
                
                if assistant_config:
                    logger.info(f"📋 Loaded Config: {assistant_config.display_name} ({assistant_config.id})")
                    call_session = CallSession(call_id=call_id, assistant_config=assistant_config, stream_sid=stream_sid)
                    
                    # Resolve ID for Webhook (Legacy/Vapi ID vs Internal)
                    # Check metadata for 'vapi_assistant_id'
                    meta = assistant_config.metadata or {}
                    assistant_id_webhook = meta.get("vapi_assistant_id", assistant_config.id)
                    
                    # Update System Prompt from Config safely
                    base_instruction = assistant_config.system_prompt or "You are a helpful assistant."
                else:
                    logger.warning("⚠️ No Assistant Config found, using defaults.")
                    base_instruction = "You are a helpful assistant."
                    assistant_id_webhook = internal_assistant_id # Fallback

                # Append Tool Usage Optimization
                # Force instant execution and suppress "planning" monologues
                # Revert to simple instruction (remove Tool Rules which might be causing silence/confusion)
                system_instruction = base_instruction

                # Emit call.started
                if assistant_id_webhook and call_id:
                    emitter = get_supabase_vapi_webhook_emitter()
                    # We fire and forget this task so it doesn't block audio
                    asyncio.create_task(
                         emitter.emit_call_started(
                            call_id=call_id,
                            assistant_id=assistant_id_webhook,
                            customer_number=customer_number,
                            created_at=call_start_time
                        )
                    )

                # --- Tool Registry Setup ---
                from src.tools.registry import ToolRegistry, ToolContext
                from src.tools.scheduling import get_open_slots_tool, book_appointment_tool
                from src.tools.telephony import transfer_call_tool
                from src.tools.schemas import GetOpenSlotsArgs, BookAppointmentArgs, TransferCallArgs

                tool_registry = ToolRegistry()
                
                # Register Scheduling Tools
                tool_registry.register(
                    name="getOpenSlots",
                    description="Checks calendar availability. Arguments: 'requestedAppointment' (ISO string like '2024-12-25T10:00:00+02:00'). Returns list of slots.",
                    args_model=GetOpenSlotsArgs,
                    side_effect=False
                )(get_open_slots_tool)

                tool_registry.register(
                    name="bookAppointment",
                    description="Books an appointment. REQUIRED: 'name', 'requestedAppointment' (exact ISO string from getOpenSlots result).",
                    args_model=BookAppointmentArgs,
                    side_effect=True
                )(book_appointment_tool)

                # Register Telephony Tools
                tool_registry.register(
                    name="transfer_call_tool",
                    description="Use this tool to transfer the caller to a real person when requested or when escalation is needed.",
                    args_model=TransferCallArgs,
                    side_effect=True
                )(transfer_call_tool)
                
                # Create Tool Context
                tool_context = ToolContext(
                    call_id=call_id,
                    twilio_call_sid=call_id, # Twilio sends callSid as callId usually
                    professional_slug=assistant_config.professional_slug if assistant_config else "unknown",
                    assistant_id_webhook=assistant_id_webhook,
                    caller_timezone=assistant_config.timezone if assistant_config else "Asia/Jerusalem",
                    state={}
                )

                # Re-Initialize Client with Tools
                # We overwrite the initial client (which was empty)
                client = GeminiLiveClient(
                    input_queue=mic_queue, 
                    output_queue=speaker_queue,
                    tool_registry=tool_registry,
                    tool_context=tool_context
                )

                # Start Gemini Session with Configured Prompt
                # Voice is static (defaults in client), ignoring config.voice_id as requested
                gemini_task = asyncio.create_task(
                    client.start(
                        system_instruction=system_instruction
                    )
                )
                
                gemini_task.set_name("Gemini_Client_Task")

                # Start Outbound Sender
                async def send_audio_to_twilio():
                    logger.info("🚀 Starting Twilio Sender Loop")
                    try:
                        while True:
                            # Get PCM24k/16k from Gemini
                            chunk = await speaker_queue.get()
                            if not chunk: continue
                            
                            # Resample to 8kHz for Twilio
                            in_rate = config.get("audio.receive_sample_rate", 24000)
                            resampled = resample_audio(chunk, in_rate, 8000)
                            
                            # Encode to mulaw
                            payload = base64.b64encode(pcm_to_mulaw(resampled)).decode("utf-8")
                            
                            # Send to Twilio
                            if stream_sid:
                                await websocket.send_json({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": payload}
                                })
                                
                    except Exception as e:
                        logger.error(f"❌ Error sending to Twilio: {e}", exc_info=True)
                        raise
                
                sender_task = asyncio.create_task(send_audio_to_twilio())
                sender_task.set_name("Twilio_Sender_Task")

            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    # Decode Base64 -> mulaw -> PCM16 (8kHz)
                    pcm_8k = mulaw_to_pcm(base64.b64decode(payload))
                    
                    # Resample 8k -> 16k (SEND_SAMPLE_RATE)
                    out_rate = 16000
                    pcm_16k = resample_audio(pcm_8k, 8000, out_rate)
                    
                    await mic_queue.put(pcm_16k)
                    
            elif event == "stop":
                logger.info("🛑 Stream Stopped")
                break
            
            elif event == "mark":
                pass
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket Disconnected")
    except Exception as e:
        logger.error(f"❌ Main Media Stream Loop Crashed: {e}", exc_info=True)
    finally:
        # Cleanup
        if gemini_task: gemini_task.cancel()
        if sender_task: sender_task.cancel()
        try:
            await websocket.close()
        except:
            pass
        
        # Emit call.ended
        if call_id and assistant_id_webhook and call_start_time:
             ended_at = datetime.now(timezone.utc)
             emitter = get_supabase_vapi_webhook_emitter()
             # Fire and forget the webhook task
             asyncio.create_task(
                 emitter.emit_call_ended(
                    call_id=call_id,
                    assistant_id=assistant_id_webhook,
                    customer_number=customer_number or "unknown",
                    created_at=call_start_time,
                    ended_at=ended_at,
                    transcript="Transcript not available yet", # Placeholder
                    ended_reason="completed" # Simplified
                )
             )
        
        logger.info("👋 Media Stream Cleanup Complete")
