import asyncio
import logging
from google import genai
from src.config.environment import config

logger = logging.getLogger(__name__)

class GeminiLiveClient:
    """Handles the connection and bidirectional communication with Gemini Live."""

    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.session = None

    async def start(self, system_instruction: str = None):
        """Connects to Gemini Live and starts send/receive loops."""
        
        config_params = {}
        
        if not config.get("gemini.use_defaults", False):
            config_params = {
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "temperature": config.get("gemini.temperature"),
                },
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": config.get("gemini.voice_name")
                        }
                    }
                }
            }
        else:
            # minimal config even for defaults
            config_params = {"response_modalities": ["AUDIO"]}

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
                logger.info("✅ Connected to Gemini Live")

                # Run send/receive loops in parallel
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

    async def send_text(self, text: str):
        """Sends a text message to the Gemini session (e.g. to trigger a response)."""
        if self.session:
            await self.session.send(input=text, end_of_turn=True)
            logger.info(f"📤 Sent text to Gemini: {text}")

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
        """Receives responses from Gemini and pushes audio to output queue."""
        try:
            while True:
                # We need to loop over receive() which yields "turns"
                # And turns yield "responses"
                # This might change based on exact SDK version, keeping consistent with POC
                
                # Note: The POC had 'turn = session.receive()', but in async context 
                # we usually iterate: 'async for response in session.receive():'
                # Let's check the POC reference carefully.
                # POC:
                # turn = session.receive()
                # async for response in turn: ...
                
                # Implementation:
                turn = self.session.receive()
                async for response in turn:
                    try:
                        sc = getattr(response, "server_content", None)
                        if not sc or not sc.model_turn:
                            continue

                        for part in sc.model_turn.parts:
                            inline = getattr(part, "inline_data", None)
                            if inline and isinstance(inline.data, (bytes, bytearray)):
                                # Audio data (PCM 24kHz)
                                await self.output_queue.put(inline.data)
                    except Exception as e:
                        logger.error(f"Error processing response part: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
