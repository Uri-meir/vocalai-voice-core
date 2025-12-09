import asyncio
import pyaudio
import logging
from src.config.environment import config

logger = logging.getLogger(__name__)

class SpeakerStream:
    """Manages the speaker output stream."""

    def __init__(self, queue: asyncio.Queue[bytes]):
        self.queue = queue
        self.pya = pyaudio.PyAudio()
        self.stream = None

    async def start(self):
        """Starts the audio player consumer loop."""
        try:
            self.stream = await asyncio.to_thread(
                self.pya.open,
                format=pyaudio.paInt16,
                channels=config.get("audio.channels"),
                rate=config.get("audio.receive_sample_rate"),
                output=True,
            )
            logger.info("🔊 Speaker stream ready")
            
            await self._play_loop()
            
        except Exception as e:
            logger.error(f"Failed to start speaker: {e}")
            raise

    async def _play_loop(self):
        """Internal loop to play audio chunks."""
        while True:
            try:
                chunk = await self.queue.get()
                if chunk is None: # Sentinel value to stop
                    break
                
                await asyncio.to_thread(self.stream.write, chunk)
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error playing audio chunk: {e}")

    def stop(self):
        """Stops the speaker stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.pya.terminate()
        logger.info("🔊 Speaker stream closed")
