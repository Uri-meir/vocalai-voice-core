import asyncio
import logging
import time
import json
import base64
import websockets
from typing import Optional, Dict, Any
from src.config.environment import config
from src.gemini.client import TurnState  # Shared TurnState for consistent gating in media_stream.py

logger = logging.getLogger(__name__)

class OpenAIModelNotFound(RuntimeError):
    """Raised when the selected Realtime model id is not available for the API key."""

class OpenAIResponseFailed(RuntimeError):
    """Raised when OpenAI returns response.done with failed status or empty output (prevents silent calls)."""

class OpenAILiveClient:
    """Handles the connection and bidirectional communication with OpenAI Realtime API."""

    def __init__(
        self, 
        input_queue: asyncio.Queue, 
        output_queue: asyncio.Queue,
        tool_registry=None,
        tool_context=None,
        termination_queue: asyncio.Queue = None
    ):
        self.api_key = config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tool_registry = tool_registry
        self.tool_context = tool_context
        self.websocket = None
        
        # 3-state turn management (mirroring Gemini client)
        self.turn_state = TurnState.USER
        self.last_model_activity_at = time.monotonic()
        self.tools_in_flight = 0
        self.termination_queue = termination_queue
        
        # User silence tracking
        self.last_user_activity_at = None
        self.user_silence_warning_level = 0
        self.user_silence_monitor_task = None
        self.call_start_time = time.monotonic()
        
        # Watchdog state
        self.playout_until = 0.0
        self.playout_started_at = 0.0
        self.model_has_spoken_this_turn = False  # Provider-neutral naming
        self.silence_accumulator_s = 0.0
        self.nudge_count_this_turn = 0
        self.max_nudges_per_turn = config.get("watchdog.max_nudges_per_turn", 2)
        self.silence_timeout_s = config.get("watchdog.silence_timeout_ms", 6000) / 1000.0
        self.watchdog_task = None
        
        # Transcript tracking
        self.current_turn_transcript = []
        self.current_user_transcript = []
        self.transcript_log = []

        # Assistant "first message" (from DB) to reuse for first watchdog nudge if needed
        self.first_message_text: Optional[str] = None
        
        # OpenAI specific state
        self.session_id = None
        self.conversation_id = None
        
        # Observability: Track bytes sent to provider
        self.audio_sent_bytes = 0
        self.last_audio_sent_log_time = 0.0
        self.audio_log_interval = 10.0  # Log every 10 seconds

    def mark_model_activity(self):
        """Update last model activity timestamp."""
        self.last_model_activity_at = time.monotonic()
        self.silence_accumulator_s = 0.0

    def mark_user_activity(self):
        """
        Called when ANY voice activity detected (VAD or RMS burst).
        Updates last activity timestamp during USER turn.
        """
        if self.turn_state == TurnState.USER:
            now = time.monotonic()
            if self.last_user_activity_at:
                silence_duration = now - self.last_user_activity_at
                logger.info(
                    f"🎤 USER ACTIVITY DETECTED - Silence timer reset "
                    f"(was {silence_duration:.1f}s silent, warning_level={self.user_silence_warning_level})"
                )
            else:
                logger.info("🎤 USER ACTIVITY DETECTED - Silence tracking initialized")
            
            self.last_user_activity_at = now
            self.user_silence_warning_level = 0

    def transition_to_user(self, reason=""):
        """Switch to USER state - mirroring Gemini client exactly."""
        old_state = self.turn_state.value if self.turn_state else "none"
        self.turn_state = TurnState.USER
        
        # Reset silence timer (bot speech doesn't count as user silence)
        # BUT preserve warning level (escalation continues unless user actually speaks)
        self.last_user_activity_at = time.monotonic()
        
        # Reset turn tracking
        self.nudge_count_this_turn = 0  # Reset for next turn
        self.silence_accumulator_s = 0.0  # Reset silence
        self.playout_until = 0.0  # Reset playout tracking
        self.model_has_spoken_this_turn = False  # Reset speech flag
        
        # Log turn change with silence tracking info
        warning_interval = config.get("turn.user_silence_warning_interval_s", 8)
        logger.info(
            f"🔄 TURN {old_state} → USER ({reason}) | "
            f"Warning Level: {self.user_silence_warning_level} | "
            f"Intervals: {warning_interval}s (3 warnings → terminate at {warning_interval * 3}s)"
        )

    def transition_to_gemini(self, reason=""):
        """Switch to GEMINI state (commit) - mirroring Gemini client exactly."""
        old_state = self.turn_state.value if self.turn_state else "none"
        
        # Save accumulated user transcript before transitioning
        full_user_transcript = "".join(self.current_user_transcript)
        if full_user_transcript:
            self.transcript_log.append({
                "turn_id": len(self.transcript_log) + 1,
                "speaker": "user",
                "timestamp": time.time(),
                "text": full_user_transcript
            })
            logger.info(f"💾 User transcript (full turn): {full_user_transcript[:100]}...")
        self.current_user_transcript = []  # Reset for next turn
        
        self.turn_state = TurnState.GEMINI
        self.mark_model_activity()  # Update activity timestamp
        self.nudge_count_this_turn = 0  # Reset nudge counter for new turn
        self.silence_accumulator_s = 0.0  # Reset silence
        self.playout_until = 0.0  # Reset playout tracking
        self.model_has_spoken_this_turn = False  # Reset speech flag (will be set when audio arrives)
        logger.info(f"🔄 TURN {old_state} → GEMINI ({reason})")

    def on_user_resumed_speaking(self, reason=""):
        """Handle user resuming speech during silence escalation."""
        logger.info(f"🎤 User resumed speaking ({reason}) - resetting warning level to 0")
        self.user_silence_warning_level = 0

    async def start(
        self, 
        system_instruction: str = None, 
        initial_text: str = None, 
        voice_name: str = None,
        model_id: str = None,
        temperature: float = None
    ):
        """
        Connect to OpenAI Realtime API and start session.
        """
        requested_model_id = model_id or config.get("openai.model_id", "gpt-realtime-2025-08-28")
        fallback_model_id = config.get("openai.fallback_model_id", "gpt-realtime-2025-08-28")
        voice = voice_name or config.get("openai.voice_name", "alloy")
        temp = temperature if temperature is not None else config.get("openai.temperature", 0.8)

        # Keep a copy for watchdog nudges (do NOT tag here; this is the actual greeting)
        self.first_message_text = (initial_text or "").strip() or None

        # Try requested model first; if not accessible, retry with fallback model.
        models_to_try = []
        for mid in (requested_model_id, fallback_model_id):
            if mid and mid not in models_to_try:
                models_to_try.append(mid)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        last_error: Exception | None = None

        for attempt_model_id in models_to_try:
            # Build WebSocket URL
            ws_url = "wss://api.openai.com/v1/realtime?model=" + attempt_model_id
            logger.info(f"Connecting to OpenAI Realtime (Model: {attempt_model_id}, Voice: {voice})...")

            try:
                async with websockets.connect(
                    ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    self.call_start_time = time.monotonic()
                    logger.info("✅ Connected to OpenAI Realtime")

                    # Add Hebrew language instruction to system prompt if configured
                    language_instruction = config.get("openai.language_instruction", "")
                    full_instructions = system_instruction or "You are a helpful assistant."
                    if language_instruction:
                        full_instructions = f"{language_instruction}\n\n{full_instructions}"
                        logger.info("🌐 Hebrew language enforcement enabled")

                    # Configure session
                    session_config = {
                        "type": "session.update",
                        "session": {
                            # Prefer audio-first ordering (some model versions are picky / buggy here)
                            "modalities": ["audio", "text"],
                            "instructions": full_instructions,
                            "voice": voice,
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                            "input_audio_transcription": {
                                "model": "whisper-1"
                            } if config.get("openai.enable_transcription", True) else None,
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500
                            },
                            "temperature": temp,
                            "max_response_output_tokens": "inf"
                        }
                    }

                    # Add tools if available
                    if self.tool_registry:
                        tools = self.tool_registry.get_openai_tools()
                        if tools:
                            session_config["session"]["tools"] = tools
                            logger.info(f"🔧 Injecting {len(tools)} tools into OpenAI session")

                    await self.websocket.send(json.dumps(session_config))

                    # Start monitoring tasks (restart per connection attempt)
                    if self.user_silence_monitor_task:
                        self.user_silence_monitor_task.cancel()
                        try:
                            await self.user_silence_monitor_task
                        except asyncio.CancelledError:
                            pass

                    self.user_silence_monitor_task = asyncio.create_task(self._user_silence_monitor_loop())
                    self.user_silence_monitor_task.set_name("User_Silence_Monitor")

                    if self.watchdog_task:
                        self.watchdog_task.cancel()
                        try:
                            await self.watchdog_task
                        except asyncio.CancelledError:
                            pass

                    self.watchdog_task = asyncio.create_task(self._turn_based_watchdog_loop())
                    self.watchdog_task.set_name("Turn_Based_Watchdog")

                    # Run send/receive loops
                    send_task = asyncio.create_task(self._send_loop(), name="OpenAI_Send_Loop")
                    recv_task = asyncio.create_task(self._receive_loop(), name="OpenAI_Receive_Loop")

                    # Send initial message if provided (await so model_not_found is detected quickly)
                    if initial_text:
                        logger.info(f"🗣️ Sending First Message: {initial_text}")
                        await self.send_text(initial_text)

                    await asyncio.gather(send_task, recv_task)
                    return

            except OpenAIModelNotFound as e:
                last_error = e
                logger.error(f"❌ OpenAI model not found / no access: {e}")
                if attempt_model_id != models_to_try[-1]:
                    logger.info(f"🔄 Retrying OpenAI with fallback model: {models_to_try[-1]}")
                continue
            except OpenAIResponseFailed as e:
                last_error = e
                logger.error(f"❌ OpenAI response failed for model={attempt_model_id}: {e}")
                if attempt_model_id != models_to_try[-1]:
                    logger.info(f"🔄 Retrying OpenAI with fallback model: {models_to_try[-1]}")
                continue
            except Exception as e:
                last_error = e
                logger.error(f"OpenAI connection error: {e}")
                raise
            finally:
                logger.info("OpenAI connection closed")

                # Cleanup (per attempt)
                if self.user_silence_monitor_task:
                    self.user_silence_monitor_task.cancel()
                    try:
                        await self.user_silence_monitor_task
                    except asyncio.CancelledError:
                        pass
                    self.user_silence_monitor_task = None

                if self.watchdog_task:
                    self.watchdog_task.cancel()
                    try:
                        await self.watchdog_task
                    except asyncio.CancelledError:
                        pass
                    self.watchdog_task = None

        # If we exhausted attempts, raise the last meaningful error so media_stream can fallback.
        if last_error:
            raise last_error
        raise RuntimeError("Failed to start OpenAI Realtime session")

    async def send_text(self, text: str, end_of_turn: bool = True):
        """Send text input to OpenAI."""
        if self.websocket:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            await self.websocket.send(json.dumps(event))
            
            # Trigger response if end_of_turn
            if end_of_turn:
                # Request audio+text output (removing invalid instructions field)
                response_event = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"]
                    }
                }
                await self.websocket.send(json.dumps(response_event))
                logger.debug(f"📤 response.create sent with audio+text modality")
            
            self.mark_model_activity()
            logger.info(f"📤 Sent text to OpenAI: {text} (eot={end_of_turn})")

    async def interrupt(self):
        """Interrupts the model generation."""
        if self.websocket:
            event = {"type": "response.cancel"}
            await self.websocket.send(json.dumps(event))
            logger.info("🛑 Sent Interrupt Signal to OpenAI")

    async def _send_loop(self):
        """Continuously reads from input queue and sends audio to OpenAI."""
        while True:
            try:
                data = await self.input_queue.get()
                if data is None:
                    break
                
                # Send audio as base64-encoded PCM16
                audio_b64 = base64.b64encode(data).decode('utf-8')
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await self.websocket.send(json.dumps(event))
                self.input_queue.task_done()
                
                # Track audio sent for observability
                self.audio_sent_bytes += len(data)
                now = time.time()
                if now - self.last_audio_sent_log_time >= self.audio_log_interval:
                    rate_kbps = (self.audio_sent_bytes / (now - (self.last_audio_sent_log_time or now) or 1)) / 1024
                    logger.info(
                        f"📊 OpenAI audio sent: {self.audio_sent_bytes / 1024:.1f}KB total "
                        f"({rate_kbps:.1f} KB/s)"
                    )
                    self.last_audio_sent_log_time = now
                
            except Exception as e:
                logger.error(f"Error sending audio to OpenAI: {e}")
                break

    async def _receive_loop(self):
        """Receives events from OpenAI and handles them."""
        debug_events = bool(config.get("openai.debug_events", False))
        try:
            async for message in self.websocket:
                try:
                    event = json.loads(message)
                    event_type = event.get("type")

                    # Debug-only event tracing (OFF by default for production)
                    if debug_events and event_type not in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"]:
                        logger.debug(f"🔍 OpenAI event: {event_type}")

                    # Debug-only full payload dump
                    if debug_events and event_type in ["response.done", "error"]:
                        logger.debug(f"🔍 Full event: {json.dumps(event, ensure_ascii=False)[:4000]}")
                    
                    # Handle session events
                    if event_type == "session.created":
                        self.session_id = event.get("session", {}).get("id")
                        logger.info(f"📝 Session created: {self.session_id}")
                    
                    elif event_type == "session.updated":
                        if debug_events:
                            logger.debug("📝 Session updated")
                    
                    # Handle audio output
                    elif event_type in ("response.audio.delta", "response.output_audio.delta"):
                        # Different SDKs / model versions may use different event names/fields.
                        audio_b64 = event.get("delta") or event.get("audio") or event.get("data")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            await self.output_queue.put(audio_bytes)
                            self.mark_model_activity()

                            if not self.model_has_spoken_this_turn:
                                self.model_has_spoken_this_turn = True
                                logger.info("🎯 Model has spoken this turn (first audio)")

                            # Commit to GEMINI state if not already
                            if self.turn_state != TurnState.GEMINI:
                                self.transition_to_gemini(reason="audio_output")
                    
                    elif event_type in ("response.audio.done", "response.output_audio.done"):
                        logger.info("🎵 Audio response completed")
                    
                    elif event_type == "response.output_item.done":
                        if debug_events:
                            logger.debug(f"📦 Output item done: {event.get('item', {}).get('type', 'unknown')}")
                    
                    # Handle transcription
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            self.current_user_transcript.append(transcript)
                            if debug_events:
                                logger.debug(f"📝 User transcript: {transcript[:100]}...")
                    
                    elif event_type == "response.audio_transcript.delta":
                        delta = event.get("delta", "")
                        if delta:
                            self.current_turn_transcript.append(delta)
                            logger.debug(f"📝 Assistant transcript delta: {delta[:50]}...")
                    
                    elif event_type == "response.audio_transcript.done":
                        transcript = event.get("transcript", "")
                        if transcript:
                            if debug_events:
                                logger.debug(f"📝 Assistant transcript complete: {transcript[:100]}...")
                    
                    # Handle function calls
                    elif event_type == "response.function_call_arguments.delta":
                        # Accumulate function call arguments
                        pass
                    
                    elif event_type == "response.function_call_arguments.done":
                        call_id = event.get("call_id")
                        name = event.get("name")
                        arguments = event.get("arguments")
                        
                        if name and arguments:
                            logger.info(f"🛠️ Function call: {name}")
                            asyncio.create_task(self._handle_function_call(call_id, name, arguments))
                    
                    # Handle response lifecycle
                    elif event_type == "response.created":
                        logger.info("🎬 Response generation started")
                    
                    elif event_type == "response.content_part.added":
                        content_part = event.get("part", {})
                        logger.debug(f"📦 Content part added: {content_part.get('type', 'unknown')}")
                    
                    elif event_type == "response.done":
                        # Quick summary to understand why we got "done" but no audio
                        resp = event.get("response", {}) or {}
                        status = resp.get("status")
                        status_details = resp.get("status_details") or {}

                        # If OpenAI reports failure, treat as fatal (prevents silent calls)
                        if status == "failed":
                            err = status_details.get("error") or {}
                            code = err.get("code") or "unknown_error"
                            msg = err.get("message") or str(err) or "OpenAI response failed"
                            logger.error(f"❌ OpenAI response.failed: {code}: {msg}")
                            if code == "model_not_found":
                                raise OpenAIModelNotFound(msg)
                            raise OpenAIResponseFailed(f"{code}: {msg}")

                        output = resp.get("output", None)
                        if output is None:
                            # Some variants nest output differently; log keys to debug schema mismatch
                            logger.warning(f"⚠️ response.done without response.output. response keys={list(resp.keys())}")
                            raise OpenAIResponseFailed("response.done missing response.output")

                        logger.info(f"📦 response.output items={len(output)}")

                        # If response completed but produced no outputs, treat as fatal (prevents silent calls)
                        # Note: do NOT fail on cancelled turns (e.g., turn_detected) where outputs may be partial.
                        if status == "completed" and len(output) == 0:
                            raise OpenAIResponseFailed("response.done completed with zero outputs")

                        # Save assistant transcript
                        full_transcript = "".join(self.current_turn_transcript)
                        if full_transcript:
                            self.transcript_log.append({
                                "turn_id": len(self.transcript_log) + 1,
                                "speaker": "assistant",
                                "timestamp": time.time(),
                                "text": full_transcript
                            })
                            logger.info(f"💾 Assistant transcript: {full_transcript[:100]}...")
                        self.current_turn_transcript = []
                        
                        logger.info("✅ OpenAI response complete")
                        self.transition_to_user(reason="response_complete")
                    
                    # Handle errors
                    elif event_type == "error":
                        error_info = event.get("error", {})
                        logger.error(f"❌ OpenAI error: {error_info}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode OpenAI event: {e}")
                except OpenAIModelNotFound:
                    raise
                except OpenAIResponseFailed:
                    raise
                except Exception as e:
                    logger.error(f"Error processing OpenAI event: {e}", exc_info=True)
                    
        except asyncio.CancelledError:
            pass
        except OpenAIModelNotFound:
            raise
        except OpenAIResponseFailed:
            raise
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def _handle_function_call(self, call_id: str, name: str, arguments_str: str):
        """Execute function and send response back to OpenAI."""
        self.tools_in_flight += 1
        
        try:
            # Parse arguments
            args = json.loads(arguments_str)
            
            logger.info(f"🛠️ Tool Call: {name}({args})")
            
            # Execute tool
            if self.tool_registry:
                result = await self.tool_registry.execute(name, args, self.tool_context)
                response_data = result.model_dump(mode='json')
            else:
                logger.error("Tool registry not initialized but tool called!")
                response_data = {"error": "Registry not initialized"}
            
            # Send function output
            output_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(response_data)
                }
            }
            await self.websocket.send(json.dumps(output_event))
            
            # Trigger response generation with audio output
            response_event = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"]
                }
            }
            await self.websocket.send(json.dumps(response_event))
            
            logger.info(f"✅ Tool '{name}' executed and response sent")
            self.mark_model_activity()
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
        finally:
            self.tools_in_flight = max(0, self.tools_in_flight - 1)

    async def _user_silence_monitor_loop(self):
        """
        Monitor continuous silence during USER turn using dual-gate (VAD + RMS).
        2-level escalation - mirroring Gemini client exactly.
        """
        logger.info("🎯 User silence monitor started (2-level escalation)")
        
        warning_interval = config.get("turn.user_silence_warning_interval_s", 8)
        message_1 = config.get("turn.silence_message_1", "סליחה לא שמעתי מה אמרתה")
        message_final = config.get("turn.silence_message_final", "עדיין אין מענה, אסיים את השיחה כעת")
        grace_period = config.get("turn.startup_grace_period_s", 2.0)
        check_playout = config.get("turn.check_playout_before_warning", True)
        
        last_log_time = 0.0
        log_interval = 1.0
        
        while True:
            try:
                await asyncio.sleep(0.2)
                
                # Only monitor during USER turn
                if self.turn_state != TurnState.USER:
                    continue
                
                # Check if activity tracking is initialized
                if not self.last_user_activity_at:
                    continue
                
                now = time.monotonic()
                silence_elapsed = now - self.last_user_activity_at
                call_duration = now - self.call_start_time
                
                # Skip if in startup grace period
                if call_duration < grace_period:
                    if now - last_log_time >= log_interval:
                        remaining_grace = grace_period - call_duration
                        logger.debug(f"🛡️ Grace period: {remaining_grace:.1f}s remaining (no silence warnings yet)")
                        last_log_time = now
                    continue
                
                # Skip if bot is still playing audio (critical fix!)
                if check_playout and now < self.playout_until:
                    playout_remaining = self.playout_until - now
                    if now - last_log_time >= log_interval:
                        logger.debug(f"🔊 Bot speaking: {playout_remaining:.1f}s remaining (silence timer paused)")
                        last_log_time = now
                    continue
                
                # Calculate time until next warning (always warning_interval from timer reset)
                time_until_next = max(0, warning_interval - silence_elapsed)
                
                # Log countdown every second (mirroring Gemini)
                if now - last_log_time >= log_interval:
                    if self.user_silence_warning_level == 0:
                        logger.info(
                            f"⏱️ USER SILENCE: {silence_elapsed:.1f}s elapsed | "
                            f"Warning in: {time_until_next:.1f}s | "
                            f"Turn: {self.turn_state.value}"
                        )
                    elif self.user_silence_warning_level == 1:
                        logger.warning(
                            f"⏱️ USER SILENCE (warned once): {silence_elapsed:.1f}s elapsed | "
                            f"Final warning in: {time_until_next:.1f}s | "
                            f"Turn: {self.turn_state.value}"
                        )
                    else:
                        logger.error(
                            f"⏱️ USER SILENCE (final): {silence_elapsed:.1f}s elapsed | "
                            f"Terminating soon | "
                            f"Turn: {self.turn_state.value}"
                        )
                    last_log_time = now
                
                # Check if warning threshold reached (mirroring Gemini exactly)
                if silence_elapsed >= warning_interval:
                    self.user_silence_warning_level += 1
                    
                    if self.user_silence_warning_level == 1:
                        # First warning
                        logger.warning(
                            f"🔔 SILENCE WARNING TRIGGERED: User silent for {silence_elapsed:.1f}s "
                            f"(threshold: {warning_interval}s)"
                        )
                        await self.send_text(f"[[WATCHDOG]] {message_1}")
                        logger.info(f"📤 Warning sent: '{message_1}' - timer reset for next interval")
                        self.last_user_activity_at = now  # Reset timer for next interval
                        
                    elif self.user_silence_warning_level == 2:
                        # Second warning = final
                        logger.error(
                            f"🛑 FINAL WARNING: User silent for {silence_elapsed:.1f}s - terminating call"
                        )
                        await self.send_text(f"[[WATCHDOG]] {message_final}")
                        logger.error(f"📤 Final warning sent: '{message_final}' - terminating")
                        
                        # Brief delay then terminate
                        await asyncio.sleep(5)
                        
                        # Signal termination
                        if self.termination_queue:
                            await self.termination_queue.put("user_silence_timeout")
                        break
                        
            except asyncio.CancelledError:
                logger.info("🛑 User silence monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in silence monitor: {e}", exc_info=True)

    async def _turn_based_watchdog_loop(self):
        """Watchdog for model silence during GEMINI turn (mirroring Gemini behavior)."""
        logger.info("🐕 Turn-based watchdog started")
        
        check_interval = 0.5
        
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                if self.turn_state != TurnState.GEMINI:
                    self.silence_accumulator_s = 0.0
                    continue
                
                if self.tools_in_flight > 0:
                    self.silence_accumulator_s = 0.0
                    continue
                
                now = time.monotonic()
                
                if now < self.playout_until:
                    self.silence_accumulator_s = 0.0
                    continue
                
                time_since_activity = now - self.last_model_activity_at
                self.silence_accumulator_s += check_interval
                
                if self.silence_accumulator_s >= self.silence_timeout_s:
                    if self.nudge_count_this_turn < self.max_nudges_per_turn:
                        self.nudge_count_this_turn += 1
                        # First nudge should repeat the assistant first_message (from DB) if available.
                        if self.nudge_count_this_turn == 1 and self.first_message_text:
                            nudge_text = self.first_message_text
                        else:
                            nudge_text = config.get("watchdog.silence_nudge_text", "המשיכי לענות עכשיו בקול")
                        
                        logger.warning(
                            f"⏰ WATCHDOG: Model silent for {self.silence_accumulator_s:.1f}s - "
                            f"sending nudge {self.nudge_count_this_turn}/{self.max_nudges_per_turn}"
                        )
                        
                        # MUST trigger spoken output. Tag as WATCHDOG so the system prompt can force "repeat exactly".
                        await self.send_text(f"[[WATCHDOG]] {nudge_text}", end_of_turn=True)
                        self.silence_accumulator_s = 0.0
                    else:
                        logger.error(
                            f"❌ WATCHDOG: Max nudges reached, model unresponsive - transitioning to USER"
                        )
                        self.transition_to_user(reason="watchdog_timeout")
                        
            except asyncio.CancelledError:
                logger.info("🛑 Watchdog cancelled")
                break
            except Exception as e:
                logger.error(f"Error in watchdog: {e}", exc_info=True)

