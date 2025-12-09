import asyncio
import pyaudio
import logging
from src.config.environment import config
from src.audio.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

class MicStream:
    """Manages the microphone input stream."""

    def __init__(self, queue: asyncio.Queue[bytes]):
        self.queue = queue
        self.pya = pyaudio.PyAudio()
        self.stream = None
        self.vad = VoiceActivityDetector()

    async def start(self):
        """Starts capturing audio from the microphone."""
        try:
            mic_info = self.pya.get_default_input_device_info()
            logger.info(f"🎙️ Using microphone: {mic_info['name']}")

            self.stream = await asyncio.to_thread(
                self.pya.open,
                format=pyaudio.paInt16,
                channels=config.get("audio.channels"),
                rate=config.get("audio.send_sample_rate"),
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=config.CHUNK_SIZE,
            )
            
            logger.info("🎙️ Mic stream started")
            await self._read_loop()
            
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            raise

    async def _read_loop(self):
        """Internal loop to read audio chunks."""
        kwargs = {"exception_on_overflow": False}
        while self.stream and self.stream.is_active():
            try:
                data = await asyncio.to_thread(self.stream.read, config.CHUNK_SIZE, **kwargs)
                
                # Check VAD
                if self.vad.is_speech(data):
                    await self.queue.put(data)
                # Else: drop the packet (send nothing), effectively muting noise
                
            except IOError as e:
                logger.warning(f"Audio overflow or read error: {e}")
                continue
            except Exception as e:
                logger.error(f"Critical error reading mic: {e}")
                break

    def stop(self):
        """Stops the microphone stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.pya.terminate()
        logger.info("🎙️ Mic stream closed")
