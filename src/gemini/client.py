import asyncio
import logging
import time
from enum import Enum
from google import genai
from google.genai import types
from src.config.environment import config

logger = logging.getLogger(__name__)

class TurnState(Enum):
    USER = "user"
    PENDING_GEMINI = "pending_gemini"
    GEMINI = "gemini"

class GeminiLiveClient:
    """Handles the connection and bidirectional communication with Gemini Live."""

    def __init__(
        self, 
        input_queue: asyncio.Queue, 
        output_queue: asyncio.Queue,
        tool_registry=None,
        tool_context=None,
        termination_queue: asyncio.Queue = None
    ):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tool_registry = tool_registry
        self.tool_context = tool_context
        self.session = None
        
        # 3-state turn management
        self.turn_state = TurnState.USER
        self.last_model_activity_at = time.monotonic()  # For watchdog
        self.tools_in_flight = 0  # Counter for concurrent tool calls
        self.termination_queue = termination_queue  # Signal call termination
        
        # User silence tracking (replaces pending timeout)
        self.last_user_activity_at = None       # Last confirmed user activity (cleaner design)
        self.user_silence_warning_level = 0     # Track escalation: 0 → 1 → 2 → 3 (terminate)
        self.user_silence_monitor_task = None   # Background monitor task
        self.call_start_time = time.monotonic() # For grace period
        
        # Watchdog state
        self.playout_until = 0.0  # Monotonic timestamp when audio playback should finish
        self.playout_started_at = 0.0  # Monotonic timestamp when FIRST chunk of turn is sent (for barge-in grace period)
        self.model_has_spoken_this_turn = False  # Track if Gemini actually spoke (not just committed)
        self.silence_accumulator_s = 0.0  # Accumulate actual silence (only increments when silent)
        self.nudge_count_this_turn = 0  # Limit nudges per Gemini turn
        self.max_nudges_per_turn = 2  # Max nudges before giving up
        self.silence_timeout_s = config.get("watchdog.silence_timeout_ms", 6000) / 1000.0
        self.watchdog_task = None  # Watchdog loop task
        
        # Transcript tracking (for call logging)
        self.current_turn_transcript = []  # Accumulate assistant transcript chunks for current turn
        self.current_user_transcript = []  # Accumulate user transcript chunks for current turn
        self.transcript_log = []  # Full conversation transcript

    def mark_model_activity(self):
        """Update last model activity timestamp (for watchdog later)."""
        self.last_model_activity_at = time.monotonic()
        self.silence_accumulator_s = 0.0  # Reset silence on any activity

    def mark_user_activity(self):
        """
        Called when ANY voice activity detected (VAD or RMS burst).
        Updates last activity timestamp during USER turn.
        """
        if self.turn_state == TurnState.USER:
            now = time.monotonic()
            # Calculate how much silence was accumulated before reset
            if self.last_user_activity_at:
                silence_duration = now - self.last_user_activity_at
                logger.info(
                    f"🎤 USER ACTIVITY DETECTED - Silence timer reset "
                    f"(was {silence_duration:.1f}s silent, warning_level={self.user_silence_warning_level})"
                )
            else:
                logger.info("🎤 USER ACTIVITY DETECTED - Silence tracking initialized")
            
            self.last_user_activity_at = now
            self.user_silence_warning_level = 0  # Reset escalation level

    def transition_to_user(self, reason=""):
        """Switch to USER state."""
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
        warning_interval = config.get("turn.user_silence_warning_interval_s", 6)
        logger.info(
            f"🔄 TURN {old_state} → USER ({reason}) | "
            f"Warning Level: {self.user_silence_warning_level} | "
            f"Intervals: {warning_interval}s (3 warnings → terminate at {warning_interval * 3}s)"
        )

    def transition_to_pending(self, reason=""):
        """Switch to PENDING_GEMINI state. (Preserved for compatibility, not actively used)"""
        old_state = self.turn_state.value if self.turn_state else "none"
        self.turn_state = TurnState.PENDING_GEMINI
        logger.info(f"⏳ TURN {old_state} → PENDING_GEMINI ({reason})")

    def transition_to_gemini(self, reason=""):
        """Switch to GEMINI state (commit)."""
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

    # DEPRECATED: No longer used with new silence detection system
    # def on_user_utterance_maybe_complete(self, reason="vad_end_or_silence"):
    #     """Called when VAD END fires - move to PENDING."""
    #     if self.turn_state != TurnState.USER:
    #         logger.debug(f"Ignoring utterance_complete (not in USER state)")
    #         return
    #     self.transition_to_pending(reason=reason)

    def on_user_resumed_speaking(self, reason="vad_start"):
        """Called when VAD START fires - cancel PENDING if active."""
        if self.turn_state == TurnState.PENDING_GEMINI:
            self.transition_to_user(reason=f"pending_cancel:{reason}")

    async def start(self, system_instruction: str = None, initial_text: str = None, voice_name: str = None, temperature: float = None):
        """Connects to Gemini Live and starts send/receive loops."""
        config_params = {
            "response_modalities": ["AUDIO"],
            "generation_config": {}
        }
        
        # Enable transcription if configured
        if config.get("gemini.enable_transcription", False):
            config_params["output_audio_transcription"] = {}
            config_params["input_audio_transcription"] = {}
            logger.info("📝 Transcription enabled for this session (input + output)")

        # Apply Temperature if provided
        if temperature is not None:
            config_params["generation_config"]["temperature"] = temperature

        # Apply Voice Config if provided
        if voice_name:
            config_params["speech_config"] = {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": voice_name
                    }
                }
            }
        
        # Inject Tools if available
        if self.tool_registry:
            declarations = self.tool_registry.get_gemini_declarations()
            if declarations:
                logger.info(f"🔧 Injecting {len(declarations)} tools into Gemini session")
                # Wrap declarations in a Tool object as required by SDK
                tool_obj = types.Tool(function_declarations=declarations)
                config_params["tools"] = [tool_obj]

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        model_id = config.get("gemini.model_id")
        logger.info(f"Connecting to Gemini Live (Model: {model_id})...")
        
        try:
            async with self.client.aio.live.connect(
                model=model_id,
                config=config_params,
            ) as session:
                self.session = session
                self.call_start_time = time.monotonic()  # Initialize call start time
                logger.info("✅ Connected to Gemini Live")

                # Send initial text if provided (fire and forget)
                if initial_text:
                    logger.info(f"🗣️ Sending First Message: {initial_text}")
                    asyncio.create_task(self.send_text(initial_text))

                # Run send/receive loops in parallel
                # Start NEW user silence monitor
                if self.user_silence_monitor_task:
                    logger.warning("⚠️ User silence monitor task already exists, cancelling old one")
                    self.user_silence_monitor_task.cancel()
                    try:
                        await self.user_silence_monitor_task
                    except asyncio.CancelledError:
                        pass
                
                self.user_silence_monitor_task = asyncio.create_task(
                    self._user_silence_monitor_loop()
                )
                self.user_silence_monitor_task.set_name("User_Silence_Monitor")
                
                # Start turn-based watchdog (single-instance)
                if self.watchdog_task:
                    logger.warning("⚠️ Watchdog task already exists, cancelling old one")
                    self.watchdog_task.cancel()
                    try:
                        await self.watchdog_task
                    except asyncio.CancelledError:
                        pass

                self.watchdog_task = asyncio.create_task(self._turn_based_watchdog_loop())
                self.watchdog_task.set_name("Turn_Based_Watchdog")
                
                # Using gather for Python 3.9 compatibility (TaskGroup is 3.11+)
                await asyncio.gather(
                    self._send_loop(),
                    self._receive_loop()
                )
                    
        except Exception as e:
            logger.error(f"Gemini connection error: {e}")
            raise
        finally:
            logger.info("Gemini connection closed")
            
            # Cleanup user silence monitor
            if self.user_silence_monitor_task:
                self.user_silence_monitor_task.cancel()
                try:
                    await self.user_silence_monitor_task
                except asyncio.CancelledError:
                    pass
                self.user_silence_monitor_task = None
            
            # Cancel watchdog task
            if self.watchdog_task:
                self.watchdog_task.cancel()
                try:
                    await self.watchdog_task
                except asyncio.CancelledError:
                    pass
                self.watchdog_task = None

    async def send_text(self, text: str, end_of_turn: bool = True):
        """Send text input to Gemini with optional end_of_turn flag."""
        if self.session:
            await self.session.send(input=text, end_of_turn=end_of_turn)
            self.mark_model_activity()
            logger.info(f"📤 Sent text to Gemini: {text} (eot={end_of_turn})")

    async def interrupt(self):
        """Interrupts the model generation explicitly."""
        if self.session:
            # Sending an empty text message with end_of_turn=True effectively stops generation
            # and invalidates the previous turn in most LLM realtime contexts.
            await self.session.send(input=" ", end_of_turn=True)
            logger.info("🛑 Sent Interrupt Signal to Gemini")

    async def _send_loop(self):
        """Continously reads from input queue and sends audio to Gemini."""
        while True:
            try:
                data = await self.input_queue.get()
                if data is None: break
                
                await self.session.send_realtime_input(
                    audio={"data": data, "mime_type": "audio/pcm"}
                )
                self.input_queue.task_done()
            except Exception as e:
                logger.error(f"Error sending audio to Gemini: {e}")
                break

    async def _receive_loop(self):
        """Receives responses from Gemini, handles interactions."""
        try:
            while True:
                # We need to loop over receive() which yields "turns"
                # And turns yield "responses"
                # NOTE: In new SDK 0.5+, receive() yields responses directly in async loop context
                # session.receive() is an async generator
                
                async for response in self.session.receive():
                    # === COMMIT LOGIC: Detect model output from USER or PENDING ===
                    # Commit to GEMINI if we're NOT already in GEMINI and model sends output
                    if self.turn_state != TurnState.GEMINI:
                        has_audio = False
                        has_text = False
                        has_tool = False
                        commit_reason = []
                        
                        # Check for tool/function call first
                        if response.tool_call and response.tool_call.function_calls:
                            has_tool = True
                            commit_reason.append("tool")
                            self.mark_model_activity()  # Mark immediately on tool call
                        
                        # Single pass through model_turn parts
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                # Check for ACTUAL audio bytes
                                inline = getattr(part, "inline_data", None)
                                if inline and isinstance(inline.data, (bytes, bytearray)) and len(inline.data) > 0:
                                    if not has_audio:
                                        has_audio = True
                                        commit_reason.append("audio")
                                    # Enqueue audio immediately (single pass!)
                                    await self.output_queue.put(inline.data)
                                    self.mark_model_activity()  # Mark on each audio chunk
                                    # Set flag for watchdog (first audio = Gemini has spoken)
                                    if not self.model_has_spoken_this_turn:
                                        self.model_has_spoken_this_turn = True
                                        logger.debug("🎯 Gemini has spoken this turn (first audio in receive_loop)")
                                
                                # Check for ACTUAL text content
                                if hasattr(part, 'text') and part.text:
                                    if not has_text:
                                        has_text = True
                                        commit_reason.append("text")
                                        self.mark_model_activity()  # Mark on text
                        
                        # Commit if ANY real output detected
                        if has_audio or has_text or has_tool:
                            # NOTE: Don't call mark_user_activity() here
                            # Gemini might be responding to a forced prompt (e.g., silence warning)
                            # Only VAD/RMS detection in media_stream should reset silence timer
                            self.transition_to_gemini(reason=f"model_started:{'+'.join(commit_reason)}")
                    
                    else:
                        # Already in GEMINI - just handle audio/text normally
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                inline = getattr(part, "inline_data", None)
                                if inline and isinstance(inline.data, (bytes, bytearray)) and len(inline.data) > 0:
                                    await self.output_queue.put(inline.data)
                                    self.mark_model_activity()  # Mark on each audio chunk
                                    # Set flag for watchdog
                                    if not self.model_has_spoken_this_turn:
                                        self.model_has_spoken_this_turn = True
                                        logger.debug("🎯 Gemini has spoken this turn (audio in GEMINI state)")
                                
                                # Mark on text too
                                if hasattr(part, 'text') and part.text:
                                    self.mark_model_activity()
                        
                        # Mark on tool call even if already in GEMINI
                        if response.tool_call and response.tool_call.function_calls:
                            self.mark_model_activity()
                    
                    # === Capture Input Transcription (User Speech) ===
                    if response.server_content and response.server_content.input_transcription:
                        user_text = response.server_content.input_transcription.text
                        # Accumulate user transcript chunks (consolidated on turn transition)
                        self.current_user_transcript.append(user_text)
                        logger.debug(f"📝 User transcript chunk: {user_text[:50]}...")
                    
                    # === Capture Output Transcription (Bot Speech) ===
                    if response.server_content and response.server_content.output_transcription:
                        transcript_text = response.server_content.output_transcription.text
                        self.current_turn_transcript.append(transcript_text)
                        logger.debug(f"📝 Assistant transcript chunk: {transcript_text[:50]}...")
                    
                    # Handle turn_complete
                    if response.server_content and response.server_content.turn_complete:
                        # Save accumulated transcript before transitioning
                        full_transcript = "".join(self.current_turn_transcript)
                        if full_transcript:
                            self.transcript_log.append({
                                "turn_id": len(self.transcript_log) + 1,
                                "speaker": "assistant",
                                "timestamp": time.time(),
                                "text": full_transcript
                            })
                            logger.info(f"💾 Assistant transcript: {full_transcript[:100]}...")
                        self.current_turn_transcript = []  # Reset for next turn
                        
                        logger.info("✅ Gemini turn_complete received")
                        self.transition_to_user(reason="turn_complete")
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            # Use fire-and-forget task to avoid blocking audio consumption
                            # We can also track them if clean shutdown is needed
                            asyncio.create_task(self._handle_tool_call(fc))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def _handle_tool_call(self, fc):
        """Executes tool and sends response back to Gemini."""
        # Increment counter at the very top
        self.tools_in_flight += 1
        
        try:
            name = fc.name
            args = fc.args
            call_id = fc.id
            
            logger.info(f"🛠️ Tool Call: {name}({args})")
            
            # Execute tool
            if self.tool_registry:
                result = await self.tool_registry.execute(name, args, self.tool_context)
                response_data = result.model_dump(mode='json')
            else:
                logger.error("Tool registry not initialized but tool called!")
                response_data = {"error": "Registry not initialized"}
            
            # Send response
            f_response = types.FunctionResponse(
                id=call_id,
                name=name,
                response=response_data
            )
            
            await self.session.send_tool_response(function_responses=[f_response])
            
            logger.info(f"✅ Tool '{name}' executed and response sent")
            
            # Mark activity after tool response sent
            self.mark_model_activity()
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
        finally:
            # Decrement counter (handles concurrent tools)
            self.tools_in_flight = max(0, self.tools_in_flight - 1)

    async def _user_silence_monitor_loop(self):
        """
        Monitor continuous silence during USER turn using dual-gate (VAD + RMS).
        2-level escalation:
        - 8s silence: Warning 1 - "סליחה לא שמעתי מה אמרתה"
        - 16s silence: Warning 2 + Terminate - "עדיין אין מענה, אסיים את השיחה כעת"
        """
        logger.info("🎯 User silence monitor started (2-level escalation)")
        
        warning_interval = config.get("turn.user_silence_warning_interval_s", 8)
        message_1 = config.get("turn.silence_message_1", "סליחה לא שמעתי מה אמרתה")
        message_final = config.get("turn.silence_message_final", "עדיין אין מענה, אסיים את השיחה כעת")
        grace_period = config.get("turn.startup_grace_period_s", 2.0)
        check_playout = config.get("turn.check_playout_before_warning", True)
        
        # Log countdown every N seconds
        last_log_time = 0.0
        log_interval = 1.0  # Log every 1 second
        
        while True:
            try:
                await asyncio.sleep(0.2)  # Check every 200ms
                
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
                
                # Skip if bot is still playing audio
                if check_playout and now < self.playout_until:
                    playout_remaining = self.playout_until - now
                    if now - last_log_time >= log_interval:
                        logger.debug(f"🔊 Bot speaking: {playout_remaining:.1f}s remaining (silence timer paused)")
                        last_log_time = now
                    continue
                
                # Calculate time until next warning (always warning_interval from timer reset)
                time_until_next = max(0, warning_interval - silence_elapsed)
                
                # Log countdown every second
                if now - last_log_time >= log_interval:
                    
                    if self.user_silence_warning_level == 0:
                        logger.info(
                            f"⏱️ USER SILENCE: {silence_elapsed:.1f}s elapsed | "
                            f"Warning in: {time_until_next:.1f}s | "
                            f"Turn: {self.turn_state.value}"
                        )
                    elif self.user_silence_warning_level == 1:
                        logger.warning(
                            f"⏱️ USER SILENCE (warned): {silence_elapsed:.1f}s elapsed | "
                            f"Terminating in: {time_until_next:.1f}s | "
                            f"Turn: {self.turn_state.value}"
                        )
                    last_log_time = now
                
                # Level 1: First warning at 8 seconds
                if silence_elapsed >= warning_interval and self.user_silence_warning_level == 0:
                    logger.warning(
                        f"🔔 SILENCE WARNING TRIGGERED: User silent for {silence_elapsed:.1f}s "
                        f"(threshold: {warning_interval}s)"
                    )
                    await self.send_text(f"Reply exactly this: {message_1}", end_of_turn=True)
                    self.user_silence_warning_level = 1
                    # Reset timer immediately so we start counting fresh for termination
                    self.last_user_activity_at = time.monotonic()
                    logger.info(f"📤 Warning sent: '{message_1}' - timer reset for next interval")
                    continue
                
                # Level 2: Final message after another 8 seconds
                if silence_elapsed >= warning_interval and self.user_silence_warning_level == 1:
                    logger.error(
                        f"☠️ SILENCE FINAL WARNING TRIGGERED: User silent for {silence_elapsed:.1f}s "
                        f"(threshold: {warning_interval}s since warning)"
                    )
                    
                    # Send final message (agent will handle call termination via tool)
                    await self.send_text(f"Reply exactly this: {message_final}", end_of_turn=True)
                    logger.error(f"📤 Final warning sent: '{message_final}' - agent will handle hangup via tool")
                    
                    # Mark as level 2 to prevent repeated messages
                    self.user_silence_warning_level = 2
                    self.last_user_activity_at = time.monotonic()
                    continue
            
            except asyncio.CancelledError:
                logger.info("🛑 User silence monitor cancelled")
                break
            except Exception as e:
                logger.error(f"User silence monitor error: {e}", exc_info=True)

    async def _turn_based_watchdog_loop(self):
        """Monitor Gemini turn for abnormal silence and send nudges."""
        logger.debug("Turn-based watchdog started")
        
        while True:
            try:
                await asyncio.sleep(1.0)  # Check every 1 second
                
                # Define now at the top of the loop
                now = time.monotonic()
                
                # === TRIPLE GATING ===
                if self.turn_state != TurnState.GEMINI:
                    self.silence_accumulator_s = 0.0  # Reset if not in GEMINI turn
                    continue  # Not Gemini's turn
                
                if self.tools_in_flight > 0:
                    self.silence_accumulator_s = 0.0  # Reset during tool execution
                    continue  # Tool executing, silence is expected
                
                # 🔑 CRITICAL: Wait until Gemini actually spoke (not just committed)
                if not self.model_has_spoken_this_turn:
                    self.silence_accumulator_s = 0.0  # Reset until speech starts
                    continue  # Text arrived but no audio yet
                
                # Check if bot is speaking (based on playout duration)
                bot_is_speaking = now < self.playout_until
                if bot_is_speaking:
                    self.silence_accumulator_s = 0.0  # Reset while speaking
                    continue  # Audio still playing
                
                # === SILENCE ACCUMULATION (only when truly silent) ===
                # Log why we're counting (all gates passed)
                playout_remaining = max(0, self.playout_until - now)
                logger.info(
                    f"🚨 Watchdog COUNTING: playout finished {-playout_remaining:.2f}s ago, "
                    f"has_spoken={self.model_has_spoken_this_turn}, "
                    f"turn={self.turn_state.value}, tools={self.tools_in_flight}"
                )
                
                self.silence_accumulator_s += 1.0  # Watchdog ticks every 1s
                
                # Show current accumulator state
                logger.info(
                    f"⏱️ Watchdog: silence={self.silence_accumulator_s:.1f}s/{self.silence_timeout_s:.1f}s, "
                    f"playout_until={playout_remaining:.2f}s, "
                    f"has_spoken={self.model_has_spoken_this_turn}, "
                    f"tools={self.tools_in_flight}, "
                    f"turn={self.turn_state.value}"
                )
                
                if self.silence_accumulator_s >= self.silence_timeout_s:
                    # Abnormal silence detected!
                    
                    # Check if we've exceeded max nudges
                    if self.nudge_count_this_turn >= self.max_nudges_per_turn:
                        logger.error(
                            f"⚠️ Max nudges ({self.max_nudges_per_turn}) reached. "
                            f"Silence: {self.silence_accumulator_s:.1f}s. Giving up."
                        )
                        # Fallback: transition to USER
                        self.transition_to_user(reason="watchdog_give_up")
                        continue
                    
                    # Log state snapshot
                    logger.warning(
                        f"🔔 WATCHDOG NUDGE #{self.nudge_count_this_turn + 1}: "
                        f"Silence={self.silence_accumulator_s:.1f}s, "
                        f"turn_state={self.turn_state.value}, "
                        f"tools_in_flight={self.tools_in_flight}"
                    )
                    
                    # Send nudge using existing send_text helper
                    nudge_text = config.get(
                        "watchdog.silence_nudge_text",
                        "המשיכי לענות עכשיו בקול"
                    )
                    await self.send_text(nudge_text)
                    
                    # Update state
                    self.nudge_count_this_turn += 1
                    self.mark_model_activity()  # Reset activity timer and silence accumulator
            
            except asyncio.CancelledError:
                logger.debug("Turn-based watchdog cancelled")
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}", exc_info=True)
