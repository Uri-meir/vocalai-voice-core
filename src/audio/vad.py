import logging
import webrtcvad
from src.config.environment import config

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """
    Wraps webrtcvad logic.
    Only active if config.get('vad.enabled') is True.
    """

    def __init__(self):
        self.enabled = config.get("vad.enabled")
        self.vad = None
        if self.enabled:
            mode = config.get("vad.mode")
            logger.info(f"🔇 VAD Enabled (Mode: {mode})")
            self.vad = webrtcvad.Vad(mode)
        else:
            logger.info("🔊 VAD Disabled (Streaming all audio)")

    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Returns True if speech is detected or if VAD is disabled.
        Returns False if silence is detected (and VAD is enabled).
        """
        if not self.enabled:
            return True
            
        try:
            return self.vad.is_speech(audio_frame, config.get("audio.send_sample_rate"))
        except Exception as e:
            logger.warning(f"VAD Error: {e}")
            # If VAD fails, default to allowing audio through
            return True
