from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import json
import base64
import asyncio
from src.telephony.audio_utils import mulaw_to_pcm, pcm_to_mulaw, resample_audio
from src.gemini.client import GeminiLiveClient, TurnState
from src.stt.soniox_client import SonioxClient
from src.config.environment import config
import time

router = APIRouter()
logger = logging.getLogger(__name__)

from src.core.session import CallSession
from src.core.assistants_repository_factory import get_assistant_repository
from src.core.events.emitter import get_supabase_vapi_webhook_emitter
from datetime import datetime, timezone

@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio Media Stream WebSocket and Bridge to Gemini."""
    # Track active connections
    import src.main as main_module
    main_module.active_connections += 1
    
    logger.info(f"🔌 WS CONNECT /twilio/media-stream headers={dict(websocket.headers)} | Active calls: {main_module.active_connections}")
    await websocket.accept()
    logger.info("✅ WS ACCEPTED")
    
    # queues for bridging
    mic_queue = asyncio.Queue()     # Twilio -> Gemini
    speaker_queue = asyncio.Queue() # Gemini -> Twilio
    termination_queue = asyncio.Queue()  # Gemini -> media_stream (for call termination)
    
    stream_sid = None
    sender_task = None
    call_session = None
    gemini_task = None
    client = None
    
    # Barge-In State
    from src.audio.vad import VoiceActivityDetector, VADState
    vad_wrapper = VoiceActivityDetector()
    barge_in_enabled = config.get("vad.barge_in_enabled", True)
    echo_guard_ms = config.get("vad.echo_guard_ms", 150)
    
    # State Machine
    state = "LISTENING"
    last_tts_send_ts = 0
    
    # Noise floor tracking (shared with RMS burst detection)
    noise_floor_rms = None  # Adaptive noise floor for RMS-based voice detection
    
    # Confirmed speech burst detector (prevents false resets from noise)
    speech_accum_ms = 0.0
    last_frame_ts = None
    rms_voice_mode = False  # Hysteresis latch
    last_burst_log_time = 0.0  # For throttled logging
    last_burst_reset_time = 0.0  # Cooldown to prevent spam resets
    
    # Barge-in state (turn-aware interruption)
    last_barge_in_at = 0.0  # Global cooldown for both VAD and RMS triggers
    ignore_speaker_audio_until = 0.0  # Prevent tail-audio leakage after barge-in
    rms_barge_in_accum_ms = 0.0  # Separate accumulator for barge-in (independent from silence RMS)
    last_barge_in_frame_ts = None  # Separate timestamp for barge-in dt calculation
    
    # Observability: Audio flow tracking
    audio_enqueued_bytes = 0  # Total bytes put into mic_queue
    last_audio_log_time = 0.0  # For throttled logging
    audio_log_interval = 10.0  # Log every 10 seconds
    
    # Helper for fire-and-forget clear messages
    async def safe_send_clear():
        try:
            if stream_sid:
                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })
        except:
            pass
    
    # Call State
    call_start_time = None
    call_id = None
    assistant_id_webhook = None
    customer_number = None
    
    # Soniox STT (Dual Channel)
    soniox_client = None
    last_soniox_text = None
    last_soniox_time = None
    fallback_timer_task = None
    last_fallback_time = None  # For cooldown tracking

    try:
        # Inbound Loop (Twilio -> Gemini)
        while True:
            # Check for termination signal (non-blocking)
            try:
                termination_reason = termination_queue.get_nowait()
                logger.error(f"🛑 TERMINATION SIGNAL RECEIVED: {termination_reason}")
                break  # Exit main loop to trigger cleanup
            except asyncio.QueueEmpty:
                pass  # No termination signal, continue normally
            
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")
            
            if event == "connected":
                logger.info(f"✅ Twilio Connected: {data}")
                
            elif event == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                call_id = data.get("start", {}).get("callSid")
                custom_params = data.get("start", {}).get("customParameters", {})
                
                # Guard against duplicate start events (can happen on reconnect/edge cases).
                # IMPORTANT: We still update stream_sid/call_id above so outbound audio uses the latest streamSid,
                # but we avoid re-initializing the provider client (which would wipe in-memory transcript state).
                if client is not None:
                    logger.warning(
                        f"⚠️ Duplicate Twilio start received; ignoring re-init. "
                        f"call_id={call_id}, stream_sid={stream_sid}"
                    )
                    continue
                
                # Internal ID used for config lookup
                internal_assistant_id = custom_params.get("assistant_id")
                
                # Attempt to get customer number from customParams (if passed from voice_hook)
                # Default to "unknown" if not available (e.g. if passed directly to TwiML without params)
                customer_number = custom_params.get("customer_number", "unknown") 

                logger.info(f"🏁 Stream Started: {stream_sid}, Assistant: {internal_assistant_id}")
                logger.info(f"📼 Recording enabled via TwiML <Start><Recording>")
                
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
                from src.tools.whatsapp import send_whatsapp_tool
                from src.tools.schemas import GetOpenSlotsArgs, BookAppointmentArgs, TransferCallArgs, SendWhatsAppArgs




                tool_registry = ToolRegistry()
                
                # Initialize cal_config and services BEFORE tool loop (needed for ToolContext later)
                cal_config = assistant_config.calendar_config if assistant_config else None
                services = cal_config.services if cal_config and cal_config.services else []
                
                # Get enabled tools from assistant metadata (with safe fallback)
                enabled_tools = assistant_config.metadata.get("enabled_tools") if assistant_config and assistant_config.metadata else None
                
                # Ensure enabled_tools is a list (handle string format from database)
                if enabled_tools is None:
                    enabled_tools = ["standard"]
                    logger.info(f"🔧 No enabled_tools metadata - defaulting to standard tools for assistant {internal_assistant_id}")
                elif isinstance(enabled_tools, str):
                    # Handle case where it's stored as string instead of JSON array
                    logger.warning(f"⚠️ enabled_tools is a string, not a list. Attempting to parse: {enabled_tools}")
                    try:
                        import ast
                        enabled_tools = ast.literal_eval(enabled_tools)
                        logger.info(f"🔧 Parsed enabled_tools: {enabled_tools}")
                    except:
                        logger.error(f"❌ Failed to parse enabled_tools string. Defaulting to ['standard']")
                        enabled_tools = ["standard"]
                else:
                    logger.info(f"🔧 Registering tools {enabled_tools} for assistant {internal_assistant_id}")
                
                # Register each enabled tool
                for tool_name in enabled_tools:
                    if tool_name == "whatsapp":
                        # Register WhatsApp tool
                        tool_registry.register(
                            name="sendWhatsApp",
                            description=(
                                "Use this tool to send a whatsapp message to the customer"
                                "Use this tool when the customer wants to use the cleaning service"
                                "You MUST ask for the customer's name before calling this tool"
                            ),
                            args_model=SendWhatsAppArgs,
                            side_effect=True,
                            timeout=10.0
                        )(send_whatsapp_tool)
                        logger.info(f"  ✅ Registered: sendWhatsApp")
                    
                    elif tool_name == "standard":
                        # Register standard tools (scheduling + transfer)
                        # Dynamic Tool Description
                        get_slots_desc = "Checks calendar availability."
                        book_appt_desc = "Books an appointment."

                        if services:
                            service_lines = [f"- {s.name} ({s.duration} mins)" for s in services]
                            service_info = "\nAvailable Services:\n" + "\n".join(service_lines) + "\nPlease providing 'duration_minutes' or 'service_name' to select the correct service."
                            
                            get_slots_desc += service_info
                            book_appt_desc += " REQUIRED: 'name', 'requestedAppointment' (exact ISO string from getOpenSlots result). " + service_info
                        else:
                            get_slots_desc += " Arguments: 'requestedAppointment' (ISO string). Returns list of slots."
                            book_appt_desc += " REQUIRED: 'name', 'requestedAppointment' (exact ISO string from getOpenSlots result)."

                        # Register Scheduling Tools
                        tool_registry.register(
                            name="getOpenSlots",
                            description=get_slots_desc,
                            args_model=GetOpenSlotsArgs,
                            side_effect=False
                        )(get_open_slots_tool)

                        tool_registry.register(
                            name="bookAppointment",
                            description=book_appt_desc,
                            args_model=BookAppointmentArgs,
                            side_effect=True
                        )(book_appointment_tool)

                        # Register Transfer Tool
                        tool_registry.register(
                            name="transfer_call_tool",
                            description="Use this tool to transfer the caller to a real person when requested or when escalation is needed.",
                            args_model=TransferCallArgs,
                            side_effect=True
                        )(transfer_call_tool)
                        
                        logger.info(f"  ✅ Registered: getOpenSlots, bookAppointment, transfer_call_tool")
                    
                    elif tool_name == "transfer":
                        # Register transfer tool only
                        tool_registry.register(
                            name="transfer_call_tool",
                            description="Use this tool to transfer the caller to a real person when requested or when escalation is needed.",
                            args_model=TransferCallArgs,
                            side_effect=True
                        )(transfer_call_tool)
                        logger.info(f"  ✅ Registered: transfer_call_tool")
                    
                    else:
                        logger.warning(f"  ⚠️ Unknown tool name: {tool_name}")
                
                # Create Tool Context
                tool_context = ToolContext(
                    call_id=call_id,
                    twilio_call_sid=call_id, 
                    professional_slug=assistant_config.professional_slug if assistant_config else "unknown",
                    assistant_id_webhook=assistant_id_webhook,
                    caller_timezone=assistant_config.timezone if assistant_config else "Asia/Jerusalem",
                    cal_username=cal_config.cal_username if cal_config else None,
                    event_type_slug=cal_config.event_type_slug if cal_config else None,
                    cal_api_key=cal_config.cal_api_key if cal_config else None,
                    services=[s.dict() for s in services], # Pass as dicts
                    event_types_by_duration=cal_config.event_types_by_duration if cal_config else {},
                    customer_number=customer_number,
                    business_phone=assistant_config.business_phone if assistant_config else None,
                    business_owner_name=assistant_config.business_owner_name if assistant_config else None,
                    state={}
                )

                # Reset VAD State
                vad_wrapper.reset()
                
                # Determine provider based on tts_model
                tts_model = (assistant_config.tts_model or "").lower() if assistant_config else ""
                use_openai = (tts_model == "openai")
                
                # Try to initialize the selected provider, fall back to Gemini on failure
                client = None
                client_task = None
                provider_name = "OpenAI" if use_openai else "Gemini"
                
                try:
                    if use_openai:
                        # Initialize OpenAI client
                        from src.openai_realtime.client import OpenAILiveClient
                        
                        logger.info(f"🤖 Using OpenAI Realtime provider for assistant {internal_assistant_id}")
                        client = OpenAILiveClient(
                            input_queue=mic_queue, 
                            output_queue=speaker_queue,
                            tool_registry=tool_registry,
                            tool_context=tool_context,
                            termination_queue=termination_queue
                        )
                        
                        # Resolve model and voice for OpenAI
                        openai_model = assistant_config.llm_model if assistant_config and assistant_config.llm_model else config.get("openai.model_id", "gpt-realtime-2025-08-28")
                        openai_voice = assistant_config.voice_id if assistant_config and assistant_config.voice_id else config.get("openai.voice_name", "alloy")
                        
                        logger.info(f"🎙️ OpenAI config: model={openai_model}, voice={openai_voice}")
                        
                        # Start OpenAI Session
                        client_task = asyncio.create_task(
                            client.start(
                                system_instruction=system_instruction,
                                initial_text=assistant_config.first_message if assistant_config and assistant_config.first_message else None,
                                voice_name=openai_voice,
                                model_id=openai_model,
                                temperature=config.get("openai.temperature", 0.8)
                            )
                        )
                        client_task.set_name("OpenAI_Client_Task")
                        
                    else:
                        # Initialize Gemini client (default)
                        client = GeminiLiveClient(
                            input_queue=mic_queue, 
                            output_queue=speaker_queue,
                            tool_registry=tool_registry,
                            tool_context=tool_context,
                            termination_queue=termination_queue
                        )

                        # Resolve Voice Name for Gemini
                        from src.core.assistant_config import ALLOWED_VOICES
                        desired_voice = assistant_config.voice_id if assistant_config and assistant_config.voice_id else None
                        
                        # Validation: Fallback to None (Default) if invalid
                        final_voice = desired_voice if desired_voice in ALLOWED_VOICES else None
                        if desired_voice and final_voice is None:
                            logger.warning(f"⚠️ Invalid voice_id '{desired_voice}'. Falling back to default.")

                        # Start Gemini Session
                        client_task = asyncio.create_task(
                            client.start(
                                system_instruction=system_instruction,
                                initial_text=f"Say exactly this: {assistant_config.first_message}" if assistant_config and assistant_config.first_message else None,
                                voice_name=final_voice,
                                temperature=0.4  # Default creativity
                            )
                        )
                        client_task.set_name("Gemini_Client_Task")
                        
                        # === Initialize Soniox STT (Dual Channel) ===
                        soniox_enabled = config.get("soniox.enabled", False)
                        if soniox_enabled and config.SONIOX_API_KEY:
                            try:
                                # Define hybrid fallback callback
                                async def on_soniox_transcript(text: str, timestamp_ms: float):
                                    """
                                    Hybrid Gating Logic:
                                    When Soniox detects a final transcript, start a fallback timer.
                                    If Gemini doesn't respond within timeout, inject the text.
                                    """
                                    nonlocal last_soniox_text, last_soniox_time, fallback_timer_task
                                    
                                    logger.info(f"🎯 Soniox callback triggered for: '{text}' (timestamp={timestamp_ms})")
                                    
                                    # Forward transcript to Gemini client for user transcript logging
                                    client.add_user_transcript(text)
                                    
                                    last_soniox_text = text
                                    last_soniox_time = time.monotonic()  # Use monotonic to match client.last_model_output_at
                                    
                                    # Cancel previous timer if exists
                                    if fallback_timer_task and not fallback_timer_task.done():
                                        fallback_timer_task.cancel()
                                    
                                    # Start fallback timer
                                    async def fallback_check():
                                        try:
                                            fallback_timeout_ms = config.get("soniox.fallback_timeout_ms", 4000)
                                            max_length = config.get("soniox.max_utterance_length", 20)
                                            max_words = config.get("soniox.max_word_count", 3)
                                            freshness_window_ms = config.get("soniox.freshness_window_ms", 1200)
                                            cooldown_ms = config.get("soniox.cooldown_ms", 800)
                                            
                                            await asyncio.sleep(fallback_timeout_ms / 1000.0)
                                        
                                            logger.info(f"⏰ Fallback timer expired for '{text}', checking conditions...")
                                        
                                            # Check if fallback is needed (robust production logic)
                                            from src.gemini.client import TurnState
                                            now_ms = time.monotonic() * 1000  # Use monotonic
                                            soniox_time_ms = last_soniox_time * 1000
                                        
                                            # Get last model output timestamp (convert to ms)
                                            last_model_output_ms = (
                                                client.last_model_output_at * 1000 
                                                if hasattr(client, 'last_model_output_at') and client.last_model_output_at
                                                else 0
                                            )
                                        
                                            # Word count check
                                            word_count = len(text.split())
                                        
                                            # Freshness check
                                            age_ms = now_ms - soniox_time_ms
                                            is_fresh = age_ms <= freshness_window_ms
                                        
                                            # Output check: no model output after Soniox detected the utterance
                                            no_model_output_after = last_model_output_ms <= soniox_time_ms
                                            
                                            # NEW GATING: Check conditions
                                            # 1. Bot speaking check (playout_until)
                                            now_sec = time.monotonic()
                                            playout_until = getattr(client, "playout_until", 0.0)
                                            bot_speaking = now_sec < playout_until
                                            
                                            # 2. Tools in flight check
                                            tools_in_flight = getattr(client, "tools_in_flight", 0)
                                            no_tools = tools_in_flight == 0
                                            
                                            # 3. Duplicate injection check (1.5s window)
                                            last_injected_text = getattr(client, "last_fallback_text", None)
                                            last_injected_ms = getattr(client, "last_fallback_injected_ms", 0)
                                            is_duplicate = (last_injected_text == text) and ((now_ms - last_injected_ms) < 1500)
                                        
                                            # Cooldown check (prevent spam)
                                            nonlocal last_fallback_time
                                            time_since_last_fallback = now_ms - (last_fallback_time * 1000 if last_fallback_time else 0)
                                            cooldown_ok = time_since_last_fallback >= cooldown_ms
                                            
                                            # Debug log (CRITICAL for diagnosing fallback issues)
                                            turn_val = client.turn_state.value if hasattr(client, 'turn_state') else 'unknown'
                                            logger.info(
                                                f"[fallback debug] turn={turn_val} "
                                                f"bot_speaking={bot_speaking} (delta={playout_until - now_sec:.2f}s) "
                                                f"no_output={no_model_output_after} "
                                                f"tools={tools_in_flight} "
                                                f"dup={is_duplicate} "
                                                f"fresh={is_fresh} "
                                            )
                                        
                                            should_fallback = (
                                                not bot_speaking           # Don't inject while bot is speaking
                                                and no_tools               # Don't inject if tool execution in progress
                                                and no_model_output_after  # Model hasn't responded to this utterance
                                                and not is_duplicate       # Don't inject duplicates
                                                and is_fresh               # Not stale
                                                and len(text) <= max_length
                                                and word_count <= max_words
                                                and cooldown_ok
                                            )
                                        
                                            if should_fallback:
                                                logger.warning(
                                                    f"🔄 Hybrid Fallback: Gemini missed short utterance. "
                                                    f"Injecting Soniox text: '{text}' "
                                                    f"(len={len(text)}, words={word_count}, age={age_ms:.0f}ms)"
                                                )
                                                
                                                # Record injection to prevent duplicates
                                                client.last_fallback_text = text
                                                client.last_fallback_injected_ms = now_ms
                                                
                                                await client.send_text(text)
                                                last_fallback_time = time.monotonic()  # Use monotonic to match now_ms
                                            else:
                                                logger.info(
                                                    f"✅ Fallback aborted for '{text}': "
                                                    f"bot_speaking={bot_speaking}, "
                                                    f"model_output={not no_model_output_after}, "
                                                    f"tools={tools_in_flight}, "
                                                    f"fresh={is_fresh}, "
                                                    f"dup={is_duplicate}"
                                                )
                                        except asyncio.CancelledError:
                                            pass  # Task cancelled (new Soniox transcript)
                                        except Exception:
                                            pass  # Call ended, connection closed
                                    
                                    fallback_timer_task = asyncio.create_task(fallback_check())
                                    fallback_timer_task.set_name("Soniox_Fallback_Timer")
                                    logger.info(f"✅ Fallback timer task created and scheduled for '{text}'")
                                
                                # Initialize Soniox client
                                soniox_client = SonioxClient(
                                    api_key=config.SONIOX_API_KEY,
                                    on_final_transcript=on_soniox_transcript,
                                    model=config.get("soniox.model", "stt-rt-preview"),
                                    language_hints=config.get("soniox.language_hints", ["he", "en"]),
                                    enable_endpoint_detection=config.get("soniox.enable_endpoint_detection", True)
                                )
                                
                                await soniox_client.connect()
                                logger.info("✅ Soniox STT initialized (Dual Channel mode)")
                                
                                # Tell Gemini client that Soniox is active, so Gemini input transcription can remain a fallback
                                try:
                                    setattr(client, "soniox_is_active", True)
                                except Exception:
                                    pass
                                
                            except Exception as e:
                                logger.error(f"❌ Failed to initialize Soniox: {e}", exc_info=True)
                                soniox_client = None
                        else:
                            logger.info("ℹ️ Soniox STT disabled (check config.soniox.enabled and SONIOX_API_KEY)")

                        
                except Exception as provider_error:
                    # Fallback to Gemini if OpenAI fails
                    if use_openai:
                        logger.error(f"❌ Failed to initialize OpenAI provider: {provider_error}")
                        logger.info("🔄 Falling back to Gemini provider")
                        
                        client = GeminiLiveClient(
                            input_queue=mic_queue, 
                            output_queue=speaker_queue,
                            tool_registry=tool_registry,
                            tool_context=tool_context,
                            termination_queue=termination_queue
                        )

                        # Resolve Voice Name for Gemini
                        from src.core.assistant_config import ALLOWED_VOICES
                        desired_voice = assistant_config.voice_id if assistant_config and assistant_config.voice_id else None
                        final_voice = desired_voice if desired_voice in ALLOWED_VOICES else None
                        if desired_voice and final_voice is None:
                            logger.warning(f"⚠️ Invalid voice_id '{desired_voice}'. Falling back to default.")

                        # Start Gemini Session
                        client_task = asyncio.create_task(
                            client.start(
                                system_instruction=system_instruction,
                                initial_text=f"Say exactly this: {assistant_config.first_message}" if assistant_config and assistant_config.first_message else None,
                                voice_name=final_voice,
                                temperature=0.4
                            )
                        )
                        client_task.set_name("Gemini_Client_Task_Fallback")
                        provider_name = "Gemini (fallback)"
                    else:
                        # Gemini init failed, re-raise
                        raise
                
                gemini_task = client_task  # Keep variable name for compatibility

                # Start Outbound Sender
                async def send_audio_to_twilio():
                    nonlocal state, last_tts_send_ts, ignore_speaker_audio_until
                    logger.info("🚀 Starting Twilio Sender Loop")
                    try:
                        while True:
                            chunk = await speaker_queue.get()
                            if not chunk: continue
                            
                            # Tail-audio suppression: Ignore chunks after barge-in to prevent Gemini leakage
                            now = time.monotonic()
                            if now < ignore_speaker_audio_until:
                                logger.debug(f"🚫 Ignoring speaker audio (tail-audio suppression for {ignore_speaker_audio_until - now:.2f}s)")
                                continue
                            
                            state = "SPEAKING"
                            in_rate = config.get("audio.receive_sample_rate", 24000)
                            resampled = resample_audio(chunk, in_rate, 8000)
                            
                            payload = base64.b64encode(pcm_to_mulaw(resampled)).decode("utf-8")
                            
                            if stream_sid:
                                await websocket.send_json({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": payload}
                                })
                                last_tts_send_ts = time.time() * 1000
                                
                                # Update client's playout tracking for watchdog
                                if client:
                                    now = time.monotonic()
                                    
                                    # Calculate audio duration: PCM int16 (2 bytes per sample), 8kHz
                                    BYTES_PER_SAMPLE = 2  # int16 PCM
                                    duration_s = len(resampled) / (8000.0 * BYTES_PER_SAMPLE)
                                    
                                    # Update playout tracking
                                    # CRITICAL: Only add buffer overlap if starting fresh.
                                    # Do NOT add 200ms per packet, or playout_until will drift by seconds!
                                    if client.playout_until > now:
                                        # Continuous playback - just append duration
                                        client.playout_until += duration_s
                                    else:
                                        # Buffer empty / starting fresh - add duration + small network jitter buffer
                                        client.playout_until = now + duration_s + 0.1
                                    
                                    # Track playout start ONCE per turn (for barge-in grace period)
                                    # Use dedicated flag to avoid coupling with receive loop
                                    if not client.playout_started_this_turn:
                                        client.playout_started_at = now
                                        client.playout_started_this_turn = True
                                        logger.info(f"🎯 Playout started this turn (first audio, duration={duration_s:.2f}s, playout_started_at={now:.2f})")
                                    
                                    # Mark that model has spoken (used by watchdog)
                                    client.model_has_spoken_this_turn = True
                                    
                                    logger.debug(
                                        f"🔊 Audio sent: {len(resampled)} bytes, "
                                        f"duration={duration_s:.2f}s, playout_until={client.playout_until - now:.2f}s from now"
                                    )
                                
                    except Exception as e:
                        logger.error(f"❌ Error sending to Twilio: {e}", exc_info=True)
                        raise
                
                sender_task = asyncio.create_task(send_audio_to_twilio())
                sender_task.set_name("Twilio_Sender_Task")
                
                # Per-call state for turn mechanism
                user_spoke_this_turn = False

            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    pcm_8k = mulaw_to_pcm(base64.b64decode(payload))
                    
                    # === 1. Calculate RMS Energy ===
                    import audioop
                    rms = audioop.rms(pcm_8k, 2)  # 2 bytes per sample (int16)
                    
                    # Frame timing for burst accumulation
                    now = time.monotonic()
                    if last_frame_ts is None:
                        dt_ms = 20.0  # Twilio frame is typically 20ms
                    else:
                        dt_ms = max(0.0, (now - last_frame_ts) * 1000.0)
                    last_frame_ts = now
                    
                    # Load barge-in config (must be before VAD try block so always defined)
                    from src.gemini.client import TurnState
                    grace_period_s = config.get("vad.barge_in_grace_period_ms", 200) / 1000.0
                    cooldown_s = config.get("vad.barge_in_cooldown_ms", 400) / 1000.0
                    respect_echo_guard = config.get("vad.barge_in_respect_echo_guard", True)
                    
                    # Echo guard check (must be before VAD block so always defined)
                    now_ms = time.time() * 1000
                    is_in_echo_guard = False
                    if state == "SPEAKING" and (now_ms - last_tts_send_ts < echo_guard_ms):
                        is_in_echo_guard = True
                    
                    # === 2. VAD Processing ===
                    if vad_wrapper.enabled:
                        try:
                            
                            vad_state = vad_wrapper.process_chunk(pcm_8k, is_echo_guard_active=is_in_echo_guard)
                            
                            # VAD START: User started speaking (HIGH CONFIDENCE)
                            if vad_state == VADState.START:
                                logger.info("🎤 VAD START - user speaking (HIGH CONFIDENCE - immediate silence reset)")
                                user_spoke_this_turn = True
                                
                                # Immediate reset - VAD is high confidence
                                if client:
                                    client.mark_user_activity()
                                    client.on_user_resumed_speaking(reason="vad_start")
                                
                                # Reset RMS burst accumulator
                                speech_accum_ms = 0.0
                                rms_voice_mode = False
                            
                            # VAD END: User stopped speaking
                            elif vad_state == VADState.END:
                                logger.info("🎤 VAD END - user finished")
                                # NOTE: Do NOT transition to PENDING anymore
                                # Just let silence accumulate naturally
                                if user_spoke_this_turn:
                                    user_spoke_this_turn = False
                            
                            # === Barge-in detection (VAD trigger - turn-aware) ===
                            # (Config loaded above, before VAD try block)
                            
                            # === GATING (turn-aware) ===
                            barge_in_vad_trigger = False
                            
                            if not client:
                                # Gate 0: Client must exist
                                pass  # Skip if client not initialized
                                
                            elif not barge_in_enabled:
                                # Gate 1: Is barge-in enabled?
                                pass  # Skip entire barge-in section
                                
                            elif client.turn_state != TurnState.GEMINI:
                                # Gate 2: Are we in GEMINI turn?
                                pass  # Only during bot speech
                                
                            elif now >= client.playout_until:
                                # Gate 3a: Is audio still playing?
                                pass  # Audio already finished
                                
                            elif now - client.playout_started_at < grace_period_s:
                                # Gate 3b: Are we past the grace period?
                                pass  # Too early (first 200ms)
                                
                            elif respect_echo_guard and is_in_echo_guard:
                                # Gate 4: Respect echo guard?
                                pass  # During echo guard window
                                
                            elif now - last_barge_in_at < cooldown_s:
                                # Gate 5: Global cooldown
                                pass  # Too soon after last barge-in
                                
                            elif vad_state == VADState.START:
                                # === ALL GATES PASSED - VAD START triggers barge-in ===
                                barge_in_vad_trigger = True
                            
                            # Execute barge-in if triggered
                            if barge_in_vad_trigger:
                                logger.warning(
                                    f"🛑 BARGE-IN (VAD) | Playout elapsed: {now - client.playout_started_at:.2f}s | "
                                    f"Remaining: {client.playout_until - now:.2f}s | "
                                    f"Silence level was: {client.user_silence_warning_level}"
                                )
                                
                                # Clear speaker queue
                                while not speaker_queue.empty():
                                    try: speaker_queue.get_nowait()
                                    except: break
                                
                                # Clear Twilio buffer
                                asyncio.create_task(safe_send_clear())
                                
                                # Interrupt Gemini
                                if hasattr(client, 'interrupt'):
                                    asyncio.create_task(client.interrupt())
                                
                                # Set tail-audio ignore window (300ms)
                                tail_ignore_s = config.get("vad.barge_in_tail_ignore_ms", 300) / 1000.0
                                ignore_speaker_audio_until = now + tail_ignore_s
                                
                                # Clear pending turn_complete and reset playout state
                                if hasattr(client, "pending_turn_end"):
                                    client.pending_turn_end = False
                                if hasattr(client, "playout_until"):
                                    client.playout_until = 0.0
                                if hasattr(client, "playout_started_this_turn"):
                                    client.playout_started_this_turn = False
                                
                                # Transition to USER turn (integrates with silence mechanism)
                                client.transition_to_user("barge-in")
                                client.user_silence_warning_level = 0  # Reset escalation (user is engaged!)
                                
                                # Update global cooldown
                                last_barge_in_at = now
                                
                        except Exception as e:
                            logger.error(f"VAD Error: {e}", exc_info=True)
                    
                    # === 3. RMS Confirmed Speech Burst Detection ===
                    # Load config
                    from src.gemini.client import TurnState
                    SNR_HIGH = config.get("turn.user_silence_snr_high", 1.35)
                    SNR_LOW = config.get("turn.user_silence_snr_low", 1.15)
                    MIN_RMS = config.get("turn.rms_absolute_minimum", 200)
                    BOOTSTRAP_MULT = config.get("turn.rms_bootstrap_multiplier", 1.3)
                    CONFIRM_MS = config.get("turn.user_silence_confirm_ms", 140)
                    DECAY_HALFLIFE_MS = config.get("turn.user_silence_decay_halflife_ms", 200)
                    MAX_ACCUM_MS = 400.0
                    
                    # Gate out bot playback/echo
                    bot_speaking = (client is not None) and (now < client.playout_until)
                    
                    if bot_speaking:
                        # While bot speaking, don't treat RMS as user speech
                        speech_accum_ms = 0.0
                        rms_voice_mode = False
                    else:
                        # Check if this frame looks like voice
                        if noise_floor_rms is None:
                            # IMPROVEMENT 1: Bootstrap noise floor when uninitialized
                            # Instead of disabling RMS detection entirely, initialize conservatively
                            noise_floor_rms = rms  # Quick bootstrap (will refine over time)
                            # Use conservative absolute threshold until floor stabilizes
                            voice_candidate = (rms > MIN_RMS * BOOTSTRAP_MULT)
                            logger.debug(f"🔧 Noise floor bootstrapped: {noise_floor_rms:.0f}")
                        else:
                            # Hysteresis: choose threshold based on current mode
                            snr_thr = SNR_LOW if rms_voice_mode else SNR_HIGH
                            voice_candidate = (rms > MIN_RMS) and (rms > noise_floor_rms * snr_thr)
                        
                        if voice_candidate:
                            rms_voice_mode = True
                            speech_accum_ms = min(MAX_ACCUM_MS, speech_accum_ms + dt_ms)
                            
                            # Log burst accumulation progress (throttled to every 0.5s)
                            if now - last_burst_log_time >= 0.5 and speech_accum_ms > 50:
                                snr = rms / noise_floor_rms if noise_floor_rms and noise_floor_rms > 0 else 0
                                logger.debug(
                                    f"🔊 RMS burst accumulating: {speech_accum_ms:.0f}ms / {CONFIRM_MS}ms "
                                    f"(RMS: {rms:.0f}, SNR: {snr:.2f}x)"
                                )
                                last_burst_log_time = now
                        else:
                            # IMPROVEMENT 3: Time-aware exponential decay (stable across packet jitter)
                            # Instead of fixed 0.65 multiplier per frame
                            decay_factor = 0.5 ** (dt_ms / DECAY_HALFLIFE_MS)
                            speech_accum_ms *= decay_factor
                            if speech_accum_ms < 20.0:
                                rms_voice_mode = False
                        
                        # Confirmed burst: only now reset silence timer
                        confirmed_speech_burst = (speech_accum_ms >= CONFIRM_MS)
                        # Add cooldown: don't reset more than once per 0.5s during continuous speech
                        cooldown_period = 0.5
                        cooldown_expired = (now - last_burst_reset_time) >= cooldown_period
                        
                        if confirmed_speech_burst and cooldown_expired and client and client.turn_state == TurnState.USER:
                            client.mark_user_activity()
                            last_burst_reset_time = now
                            snr = rms / noise_floor_rms if noise_floor_rms and noise_floor_rms > 0 else 0
                            logger.info(
                                f"🔊 RMS: Confirmed speech burst detected (MEDIUM CONFIDENCE - silence reset) | "
                                f"Burst: {speech_accum_ms:.0f}ms | RMS: {rms:.0f} | "
                                f"Floor: {noise_floor_rms:.0f} | SNR: {snr:.2f}x"
                            )
                            # Reset accumulator to prevent spam resets
                            speech_accum_ms = 0.0
                            
                            # FUTURE ENHANCEMENT: Add 2-out-of-3 check for ultra-robustness
                            # Require burst AND (recent_vad_start OR speech_like_crest_factor)
                            # This would further reduce false positives in extreme noise
                    
                    # === 3b. RMS Burst Barge-In (dual-gate alternative to VAD) ===
                    use_rms_barge_in = not config.get("vad.barge_in_vad_only", False)
                    rms_burst_threshold = config.get("vad.barge_in_rms_burst_ms", 100)
                    
                    # Only accumulate RMS burst when ALL barge-in gates pass
                    # This prevents false accumulation during echo guard, grace period, etc.
                    if not client:
                        rms_barge_in_accum_ms = 0.0
                    elif not barge_in_enabled:
                        rms_barge_in_accum_ms = 0.0
                    elif not use_rms_barge_in:
                        rms_barge_in_accum_ms = 0.0
                    elif client.turn_state != TurnState.GEMINI:
                        rms_barge_in_accum_ms = 0.0
                    elif now >= client.playout_until:
                        rms_barge_in_accum_ms = 0.0
                    elif now - client.playout_started_at < grace_period_s:
                        rms_barge_in_accum_ms = 0.0
                    elif respect_echo_guard and is_in_echo_guard:
                        rms_barge_in_accum_ms = 0.0
                    elif now - last_barge_in_at < cooldown_s:
                        rms_barge_in_accum_ms = 0.0
                    else:
                        # All gates passed - safe to accumulate RMS for barge-in
                        # Use separate timestamp to avoid conflicts with silence RMS dt
                        dt_barge_ms = 20.0 if last_barge_in_frame_ts is None else max(0.0, (now - last_barge_in_frame_ts) * 1000.0)
                        last_barge_in_frame_ts = now
                        
                        # Same voice detection as silence mechanism
                        if noise_floor_rms is None:
                            voice_candidate_barge = False
                        else:
                            SNR_HIGH_BARGE = config.get("turn.user_silence_snr_high", 1.5)
                            snr_thr_barge = SNR_HIGH_BARGE  # Use high threshold for barge-in (no hysteresis)
                            MIN_RMS_BARGE = config.get("turn.rms_absolute_minimum", 250)
                            voice_candidate_barge = (rms > MIN_RMS_BARGE) and (rms > noise_floor_rms * snr_thr_barge)
                        
                        if voice_candidate_barge:
                            rms_barge_in_accum_ms = min(400.0, rms_barge_in_accum_ms + dt_barge_ms)
                        else:
                            # Decay quickly (no hysteresis for barge-in)
                            rms_barge_in_accum_ms *= 0.5
                        
                        # Check if threshold reached
                        if rms_barge_in_accum_ms >= rms_burst_threshold:
                            snr = rms / noise_floor_rms if noise_floor_rms and noise_floor_rms > 0 else 0
                            logger.warning(
                                f"🛑 BARGE-IN (RMS) | Burst: {rms_barge_in_accum_ms:.0f}ms | "
                                f"RMS: {rms:.0f} | SNR: {snr:.2f}x | "
                                f"Silence level was: {client.user_silence_warning_level}"
                            )
                            
                            # Same interrupt logic as VAD barge-in
                            while not speaker_queue.empty():
                                try: speaker_queue.get_nowait()
                                except: break
                            asyncio.create_task(safe_send_clear())
                            if hasattr(client, 'interrupt'):
                                asyncio.create_task(client.interrupt())
                            
                            tail_ignore_s = config.get("vad.barge_in_tail_ignore_ms", 300) / 1000.0
                            ignore_speaker_audio_until = now + tail_ignore_s
                            
                            # Clear pending turn_complete and reset playout state
                            if hasattr(client, "pending_turn_end"):
                                client.pending_turn_end = False
                            if hasattr(client, "playout_until"):
                                client.playout_until = 0.0
                            if hasattr(client, "playout_started_this_turn"):
                                client.playout_started_this_turn = False
                            
                            client.transition_to_user("barge-in-rms")
                            client.user_silence_warning_level = 0  # Reset escalation
                            
                            last_barge_in_at = now
                            rms_barge_in_accum_ms = 0.0
                    
                    # === 4. Update Noise Floor ===
                    # Update noise floor only during confirmed silence
                    should_update_noise_floor = (
                        not bot_speaking
                        and not rms_voice_mode
                        and speech_accum_ms < 20.0
                    )
                    
                    if should_update_noise_floor:
                        if noise_floor_rms is None:
                            noise_floor_rms = rms
                        else:
                            noise_floor_rms = 0.98 * noise_floor_rms + 0.02 * rms
                    
                    # === 5. Forward Audio to Provider ===
                    # Drop interruptions during the FIRST bot turn (intro)
                    # This ensures the first user turn starts cleanly after the welcome message
                    drop_audio = (client is not None and getattr(client, "turn_count", 1) == 0 and client.turn_state == TurnState.GEMINI)

                    if drop_audio:
                        # Log once per burst to avoid spam
                        if rms_voice_mode and now - last_audio_log_time >= 2.0:
                            logger.info("🚫 Dropping user audio during intro (Turn 0 suppression)")
                            last_audio_log_time = now # Throttle logs
                    else:
                        out_rate = 16000
                        pcm_16k = resample_audio(pcm_8k, 8000, out_rate)
                        await mic_queue.put(pcm_16k)
                        
                        # Forward to Soniox (Dual Channel)
                        # Note: We send to Soniox even if dropped for Gemini to keep transcript logs,
                        # but fallback logic will be blocked by bot_speaking=True during intro.
                        if soniox_client and soniox_client.is_connected:
                            await soniox_client.send_audio(pcm_16k)
                    
                    # Track audio flow for observability
                    audio_enqueued_bytes += len(pcm_16k)
                    now_log = time.time()
                    if now_log - last_audio_log_time >= audio_log_interval:
                        logger.info(
                            f"📊 Audio flow: {audio_enqueued_bytes / 1024:.1f}KB enqueued to mic_queue "
                            f"({audio_enqueued_bytes / (now_log - (last_audio_log_time or now_log) or 1) / 1024:.1f} KB/s)"
                        )
                        last_audio_log_time = now_log
                    
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
             
             # Prepare transcript (JSON string)
             transcript_str = ""
             if client:
                 # Flush any final/interrupted turn data
                 transcript_data = client.get_transcript()
                 if transcript_data:
                     transcript_str = json.dumps(transcript_data, ensure_ascii=False)
             
             emitter = get_supabase_vapi_webhook_emitter()
             asyncio.create_task(
                 emitter.emit_call_ended(
                    call_id=call_id,
                    assistant_id=assistant_id_webhook,
                    customer_number=customer_number or "unknown",
                    created_at=call_start_time,
                    ended_at=ended_at,
                    transcript=transcript_str,
                    ended_reason="completed"
                )
              )
         
        # Decrement active connections
        main_module.active_connections -= 1
        logger.info(f"👋 Media Stream Cleanup Complete | Active calls: {main_module.active_connections}")
