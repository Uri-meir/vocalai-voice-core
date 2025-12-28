import logging
import onnxruntime
import numpy as np
import os
from src.config.environment import config
import time
from enum import Enum
from collections import deque
import math

logger = logging.getLogger(__name__)

# Global Singleton for ONNX Session to avoid reloading per call
_GLOBAL_ORT_SESSION = None

class VADState(Enum):
    SILENCE = 1
    START = 2
    SPEAKING = 3
    END = 4

class VoiceActivityDetector:
    """
    Silero VAD (ONNX) Implementation for Vapi-Level Robustness.
    Features:
    - Neural VAD (Silero v4/v5, 8kHz native)
    - Dual-Gate: Neural Probability + SNR Gate
    - Stateful Inference (RNN + Context)
    - Soft Echo Guard
    """

    def __init__(self):
        global _GLOBAL_ORT_SESSION
        
        self.enabled = config.get("vad.enabled", True)
        self.backend = config.get("vad.backend", "silero")
        self.session = None
        self.buffer = bytearray()
        self.last_error_time = 0
        
        # Audio Config & Framing
        self.backend_is_silero = (self.backend == "silero")
        if self.backend_is_silero:
            self.sample_rate = config.get("vad.silero_sample_rate", 8000)
            self.frame_ms = config.get("vad.silero_frame_ms", 32)
            # Silero ONNX (8k) requires exactly 256 samples
            self.frame_samples = 256
        else:
            self.sample_rate = config.get("vad.sample_rate", 8000)
            self.frame_ms = config.get("vad.frame_ms", 20)
            self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        self.frame_bytes = self.frame_samples * 2
        
        # Context Management
        self.context_size = 32 if self.sample_rate == 8000 else 64
        self.context = np.zeros(self.context_size, dtype=np.float32)
        
        # ONNX Metadata (determined dynamically)
        self.input_names = {} 
        self.output_names = []
        
        # State Tensors
        # V5 uses 'state' [2, 1, 128]
        # V4 uses 'h', 'c' [2, 1, 64] each
        self._state = None 
        self._h = None
        self._c = None
        self._state_shape = (2, 1, 128) # V5 Default

        # Configurable Thresholds
        self.threshold_normal = config.get("vad.silero_threshold", 0.5)
        self.threshold_echo = config.get("vad.silero_echo_threshold", 0.75)
        # Use ceil for safety
        self.min_speech_frames = max(1, math.ceil(config.get("vad.silero_min_speech_ms", 64) / self.frame_ms))
        
        # Legacy Logic (SNR, Guard, Flow)
        self.snr_factor = config.get("vad.snr_factor", 2.5)
        self.noise_window_ms = config.get("vad.noise_floor_window_ms", 1500)
        self.update_noise_in_echo = config.get("vad.update_noise_floor_in_echo_guard", False)
        self.freeze_noise_while_speaking = config.get("vad.freeze_noise_floor_while_speaking", True)
        
        self.echo_guard_extra_frames = config.get("vad.echo_guard_extra_frames", 2)
        self.end_silence_ms = config.get("vad.end_silence_ms", 300)
        self.speech_end_frames = int(self.end_silence_ms / self.frame_ms)
        self.barge_in_cooldown_ms = config.get("vad.barge_in_cooldown_ms", 400)

        # Energy History
        self.energy_history = deque(maxlen=int(self.noise_window_ms / self.frame_ms))
        self.current_noise_floor = 100.0

        # State Machine
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.current_state = VADState.SILENCE
        self.last_triggered_ts = 0

        if self.enabled and self.backend_is_silero:
            # Use Global Session if available, or load it
            if _GLOBAL_ORT_SESSION is not None:
                self.session = _GLOBAL_ORT_SESSION
            else:
                model_path_rel = config.get("vad.silero_model_path", "src/audio/models/silero_vad.onnx")
                model_path = os.path.join(os.getcwd(), model_path_rel)
                try:
                    opts = onnxruntime.SessionOptions()
                    opts.intra_op_num_threads = config.get("vad.onnx_intra_op_threads", 1)
                    opts.inter_op_num_threads = config.get("vad.onnx_inter_op_threads", 1)
                    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    self.session = onnxruntime.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
                    _GLOBAL_ORT_SESSION = self.session
                    logger.info(f"🧠 Silero VAD Loaded (ONNX 8k, 32ms) [Global]. Threshold: {self.threshold_normal}")
                except Exception as e:
                    logger.error(f"❌ Failed to load Silero VAD: {e}")
                    self.enabled = False
            
            # Introspect Model Inputs/Outputs
            if self.session:
                self._introspect_model()
                self._reset_state_tensors()

    def _introspect_model(self):
        """Dynamically determines input names and shapes from the ONNX model."""
        try:
            inputs = self.session.get_inputs()
            self.input_names = {i.name: i.name for i in inputs}
            
            # Check for V5 'state'
            state_input = next((i for i in inputs if 'state' in i.name), None)
            
            if state_input:
                # V5 Signature detected
                self.input_names['state'] = state_input.name
                self.input_names['input'] = next(i.name for i in inputs if 'input' in i.name)
                self.input_names['sr'] = next(i.name for i in inputs if 'sr' in i.name)
                
                # Update shape if possible (handle dynamic axes)
                shape = state_input.shape
                # V5 is typically [2, 1, 128]
                self._state_shape = tuple([d if isinstance(d, int) else 1 for d in shape])
                logger.info(f"🔎 ONNX V5 Detected: state={self._state_shape}")
            else:
                # V4 Signature (h, c)
                self.input_names['state'] = None # Flag
                self.input_names['input'] = inputs[0].name
                self.input_names['sr'] = inputs[1].name
                self.input_names['h'] = inputs[2].name
                self.input_names['c'] = inputs[3].name
                logger.info("🔎 ONNX V4 Detected (h, c split)")

            self.output_names = [o.name for o in self.session.get_outputs()]
            
        except Exception as e:
            logger.error(f"⚠️ ONNX Introspection Failed, defaulting to V5: {e}")
            # Default to V5 as that's what we likely have
            self.input_names = {'input': 'input', 'state': 'state', 'sr': 'sr'}
            self.output_names = ['output', 'stateN']
            self._state_shape = (2, 1, 128)

    def _reset_state_tensors(self):
        """Allocates zeroed state tensors."""
        if 'state' in self.input_names and self.input_names['state']:
            self._state = np.zeros(self._state_shape, dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        """Resets conversational state and RNN tensors."""
        self.buffer.clear()
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.current_state = VADState.SILENCE
        
        # Reset Neural State
        if self.backend_is_silero:
            self._reset_state_tensors()
            self.context = np.zeros(self.context_size, dtype=np.float32)

    def _calculate_rms(self, int16_samples: np.ndarray) -> float:
        """Calculates Root Mean Square energy directly from int16."""
        if len(int16_samples) == 0: return 0.0
        # Cast to float64 for accumulation to avoid overflow
        return np.sqrt(np.mean(int16_samples.astype(np.float64)**2))

    def _update_noise_floor(self, rms: float, is_echo_guard_active: bool, prob: float):
        """Updates rolling noise floor."""
        # Policy:
        # 1. Freeze if Neural Prob suggests speech (prevent self-adaptation)
        if prob > 0.1:
            return

        # 2. Freeze if Speaking
        if self.freeze_noise_while_speaking and self.current_state == VADState.SPEAKING:
            return
        
        # 3. Freeze if Echo Guard (Configurable)
        if not self.update_noise_in_echo and is_echo_guard_active:
            return
            
        # 4. Spike Protection: Don't learn from sudden loud sounds
        if rms > self.current_noise_floor * 4.0:
            return

        self.energy_history.append(rms)
        if len(self.energy_history) > 0:
            self.current_noise_floor = np.median(self.energy_history)
            self.current_noise_floor = max(self.current_noise_floor, 20.0)

    def process_chunk(self, chunk: bytes, is_echo_guard_active: bool = False) -> VADState:
        """
        Processes audio chunk with Silero VAD (native 8kHz, 32ms frames).
        """
        if not self.enabled or (self.backend_is_silero and self.session is None):
            return VADState.SILENCE

        self.buffer.extend(chunk)
        result_state = VADState.SILENCE
        
        # Process in 32ms frames
        # 32ms @ 8000Hz = 256 samples (512 bytes)
        while len(self.buffer) >= self.frame_bytes:
            # Extract Raw Frame (8k)
            raw_bytes = self.buffer[:self.frame_bytes]
            del self.buffer[:self.frame_bytes]
            
            # Int16 samples (8k)
            int16_samples_8k = np.frombuffer(raw_bytes, dtype=np.int16)

            # 1. Calculate RMS (On original 8k)
            rms = self._calculate_rms(int16_samples_8k)
            
            # 2. ONNX Inference First (Need Prob for Noise Update)
            prob = 0.0
            if self.backend_is_silero:
                # Convert to Float32 [-1, 1] - Native 8kHz
                float_samples = int16_samples_8k.astype(np.float32) / 32768.0
                
                # Context + Frame Concatenation
                x = np.concatenate([self.context, float_samples])
                ort_input_audio = x[-self.frame_samples:].astype(np.float32) # 256 samples
                self.context = x[-self.context_size:]
                
                ort_input_audio = ort_input_audio[np.newaxis, :]
                ort_sr = np.array([self.sample_rate], dtype=np.int64) # 8000
                
                try:
                    # Determine V5 or V4 inputs
                    if 'state' in self.input_names and self.input_names['state']:
                        # V5 Logic
                        if self._state is None: self._reset_state_tensors()
                        
                        ort_inputs = {
                            self.input_names['input']: ort_input_audio,
                            self.input_names['sr']: ort_sr,
                            self.input_names['state']: self._state
                        }
                        outs = self.session.run(self.output_names, ort_inputs)
                        prob = outs[0][0][0]
                        self._state = outs[1]
                    else:
                        # V4 Logic
                        if self._h is None: self._reset_state_tensors()
                        
                        ort_inputs = {
                            self.input_names['input']: ort_input_audio,
                            self.input_names['sr']: ort_sr,
                            self.input_names['h']: self._h,
                            self.input_names['c']: self._c
                        }
                        outs = self.session.run(self.output_names, ort_inputs)
                        prob = outs[0][0][0]
                        self._h = outs[1]
                        self._c = outs[2]
                    
                    # Phase 2: Store for external access
                    self._last_prob = float(prob)
                    
                except Exception as e:
                    cur_time = time.time()
                    if cur_time - self.last_error_time > 5.0:
                         logger.warning(f"Silero Error: {e}")
                         self.last_error_time = cur_time
                    prob = 0.0
            
            # 3. Update Noise Floor (Now that we have Prob)
            self._update_noise_floor(rms, is_echo_guard_active, prob)

            # 4. Decisions
            
            # Neural Gate
            current_threshold = self.threshold_echo if is_echo_guard_active else self.threshold_normal
            is_neural_speech = prob > current_threshold
            
            # SNR Gate
            threshold_rms = self.current_noise_floor * self.snr_factor
            is_snr_speech = rms > threshold_rms
            
            # Combined
            is_valid_speech = is_neural_speech and is_snr_speech
            
            if is_valid_speech:
                self.consecutive_speech_frames += 1
                self.consecutive_silence_frames = 0
            else:
                self.consecutive_silence_frames += 1
                self.consecutive_speech_frames = 0
            
            # 5. State Machine
            if self.current_state == VADState.SILENCE:
                required_frames = self.min_speech_frames
                if is_echo_guard_active:
                    required_frames += self.echo_guard_extra_frames
                
                if self.consecutive_speech_frames >= required_frames:
                    now_ms = time.time() * 1000
                    if now_ms - self.last_triggered_ts > self.barge_in_cooldown_ms:
                        self.current_state = VADState.SPEAKING
                        self.last_triggered_ts = now_ms
                        result_state = VADState.START
                    else:
                        self.consecutive_speech_frames = 0
                        
            elif self.current_state == VADState.SPEAKING:
                if self.consecutive_silence_frames >= self.speech_end_frames:
                    self.current_state = VADState.SILENCE
                    # Reset Neural State on END to prevent drift
                    if self.backend_is_silero:
                        self._reset_state_tensors()
                        
                    if result_state != VADState.START:
                        result_state = VADState.END
        
        return result_state
    
    # Phase 2: Helper methods for epoch cancellation
    def get_current_probability(self) -> float:
        """Returns the last Silero probability (0.0-1.0)."""
        return getattr(self, '_last_prob', 0.0)
    
    def get_speech_duration_ms(self) -> int:
        """Returns duration of current speech segment in milliseconds."""
        return self.consecutive_speech_frames * self.frame_ms
    
    def is_hard_interrupt(self, threshold_ms: int = 900, confidence: float = 0.7) -> bool:
        """
        Detects if current speech is a 'hard interrupt' based on:
        - Duration >= threshold_ms
        - Probability >= confidence
        
        Returns True only if BOTH conditions are met.
        """
        duration = self.get_speech_duration_ms()
        prob = self.get_current_probability()
        return duration >= threshold_ms and prob >= confidence
