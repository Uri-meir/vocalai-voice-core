from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import json
import base64
import asyncio
from src.telephony.audio_utils import mulaw_to_pcm, pcm_to_mulaw, resample_audio
from src.gemini.client import GeminiLiveClient
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
    logger.info(f"🔌 WS CONNECT /twilio/media-stream headers={dict(websocket.headers)}")
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
                
                cal_config = assistant_config.calendar_config if assistant_config else None
                services = cal_config.services if cal_config and cal_config.services else []

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
                
                # Re-Initialize Client with Tools
                client = GeminiLiveClient(
                    input_queue=mic_queue, 
                    output_queue=speaker_queue,
                    tool_registry=tool_registry,
                    tool_context=tool_context,
                    termination_queue=termination_queue
                )

                # Resolve Voice Name
                ALLOWED_VOICES = ["Puck", "Charon", "Aoede", "Fenrir", "Kore"]
                desired_voice = assistant_config.voice_id if assistant_config and assistant_config.voice_id else None
                
                # Validation: Fallback to None (Default) if invalid
                final_voice = desired_voice if desired_voice in ALLOWED_VOICES else None
                if desired_voice and final_voice is None:
                    logger.warning(f"⚠️ Invalid voice_id '{desired_voice}'. Falling back to default.")

                # Start Gemini Session
                gemini_task = asyncio.create_task(
                    client.start(
                        system_instruction=system_instruction,
                        initial_text=f"Say exactly this: {assistant_config.first_message}" if assistant_config and assistant_config.first_message else None,
                        voice_name=final_voice,
                        temperature=0.4  # Default creativity
                    )
                )
                
                gemini_task.set_name("Gemini_Client_Task")

                # Start Outbound Sender
                async def send_audio_to_twilio():
                    nonlocal state, last_tts_send_ts
                    logger.info("🚀 Starting Twilio Sender Loop")
                    try:
                        while True:
                            chunk = await speaker_queue.get()
                            if not chunk: continue
                            
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
                                    
                                    # Calculate audio duration: μ-law, 8kHz, 1 byte per sample
                                    duration_s = len(resampled) / 8000.0
                                    
                                    # Extend playout window (handles bursty delivery)
                                    client.playout_until = max(client.playout_until, now) + duration_s + 0.2
                                    
                                    # Mark that Gemini has spoken
                                    if not client.gemini_has_spoken_this_turn:
                                        logger.info(f"🎯 Gemini has spoken this turn (first audio, duration={duration_s:.2f}s)")
                                    client.gemini_has_spoken_this_turn = True
                                    
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
                    
                    # === 2. VAD Processing ===
                    if vad_wrapper.enabled:
                        try:
                            # Echo guard check
                            now_ms = time.time() * 1000
                            is_in_echo_guard = False
                            if state == "SPEAKING" and (now_ms - last_tts_send_ts < echo_guard_ms):
                                is_in_echo_guard = True
                            
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
                            
                            # Barge-in handling (existing behavior)
                            if barge_in_enabled and vad_state == VADState.START:
                                logger.info("🛑 Barge-In Triggered")
                                state = "INTERRUPTING"
                                
                                # Clear speaker queue
                                while not speaker_queue.empty():
                                    try: speaker_queue.get_nowait()
                                    except: break
                                
                                # Clear Twilio buffer
                                asyncio.create_task(safe_send_clear())
                                
                                # Interrupt Gemini
                                if hasattr(client, 'interrupt'):
                                    asyncio.create_task(client.interrupt())
                                else:
                                    asyncio.create_task(client.send_text(" "))
                                
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
                    
                    # === 5. Forward Audio to Gemini (Existing) ===
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
             asyncio.create_task(
                 emitter.emit_call_ended(
                    call_id=call_id,
                    assistant_id=assistant_id_webhook,
                    customer_number=customer_number or "unknown",
                    created_at=call_start_time,
                    ended_at=ended_at,
                    transcript="Transcript not available yet",
                    ended_reason="completed"
                )
             )
        
        logger.info("👋 Media Stream Cleanup Complete")
