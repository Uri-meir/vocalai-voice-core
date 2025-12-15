import asyncio
import logging
from google import genai
from google.genai import types
from src.config.environment import config

logger = logging.getLogger(__name__)

class GeminiLiveClient:
    """Handles the connection and bidirectional communication with Gemini Live."""

    def __init__(
        self, 
        input_queue: asyncio.Queue, 
        output_queue: asyncio.Queue,
        tool_registry=None,
        tool_context=None
    ):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tool_registry = tool_registry
        self.tool_context = tool_context
        self.session = None

    async def start(self, system_instruction: str = None, initial_text: str = None):
        """Connects to Gemini Live and starts send/receive loops."""
        config_params = {
            "generation_config": {
                "response_modalities": ["AUDIO"],
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

        if not config.get("gemini.use_defaults", False):
            # Only add speech_config if NOT using defaults
            pass # Voice config logic removed for simplicity as we stick to 2.5 defaults or add it back properly if needed
            # For now, keeping it minimal to avoid regressions:
            pass 
        else:
             pass

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

                # Send initial text if provided (fire and forget)
                if initial_text:
                    logger.info(f"🗣️ Sending First Message: {initial_text}")
                    asyncio.create_task(self.send_text(initial_text))

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
        """Receives responses from Gemini, handles interactions."""
        try:
            while True:
                # We need to loop over receive() which yields "turns"
                # And turns yield "responses"
                # NOTE: In new SDK 0.5+, receive() yields responses directly in async loop context
                # session.receive() is an async generator
                
                async for response in self.session.receive():
                    # 1. Handle Audio (Priority)
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            inline = getattr(part, "inline_data", None)
                            if inline and isinstance(inline.data, (bytes, bytearray)):
                                # Audio data (PCM 24kHz)
                                await self.output_queue.put(inline.data)

                    # 2. Handle Tool Calls (Async)
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
        try:
            name = fc.name
            args = fc.args
            call_id = fc.id
            
            logger.info(f"🛠️ Tool Call: {name}({args})")
            
            # Execute
            if self.tool_registry:
                # Returns ToolResult
                result = await self.tool_registry.execute(name, args, self.tool_context)
                
                # Serialize full envelope (success, data, error) for model clarity
                response_data = result.model_dump(mode='json')
            else:
                logger.error("Tool registry not initialized but tool called!")
                response_data = {"error": "Registry not initialized"}

            # Send Response
            f_response = types.FunctionResponse(
                id=call_id,
                name=name,
                response=response_data 
            )
            
            # logger.info(f"📤 Sending Tool Response: {response_data}")
            await self.session.send_tool_response(function_responses=[f_response])
            
        except Exception as e:
            logger.error(f"❌ Error handling tool call {fc.name}: {e}", exc_info=True)
