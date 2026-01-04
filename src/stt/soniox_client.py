"""
Soniox Real-Time Speech-to-Text Client
WebSocket-based streaming transcription for Hebrew/English
"""
import asyncio
import json
import logging
import time
from typing import Optional, Callable
import websockets

logger = logging.getLogger(__name__)

class SonioxClient:
    """
    Real-time STT client for Soniox WebSocket API.
    Handles config handshake, audio streaming, and token-based transcript parsing.
    """
    
    def __init__(
        self,
        api_key: str,
        on_final_transcript: Optional[Callable[[str, float], None]] = None,
        model: str = "stt-rt-preview",
        language_hints: list = None,
        enable_endpoint_detection: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.language_hints = language_hints or ["he", "en"]
        self.enable_endpoint_detection = enable_endpoint_detection
        self.on_final_transcript = on_final_transcript
        
        self.ws = None
        self.is_connected = False
        self.receive_task = None
        
        # Token accumulation for final transcripts
        self.current_tokens = []
        
    async def connect(self):
        """
        Establish WebSocket connection and send initial config.
        """
        endpoint = "wss://stt-rt.soniox.com/transcribe-websocket"
        
        try:
            self.ws = await websockets.connect(endpoint)
            logger.info(f"🎙️ Connected to Soniox STT: {endpoint}")
            
            # Send initial configuration (text frame)
            config = {
                "api_key": self.api_key,
                "model": self.model,
                "audio_format": "pcm_s16le",
                "num_channels": 1,
                "sample_rate": 16000,
                "language_hints": self.language_hints,
                "enable_endpoint_detection": self.enable_endpoint_detection
            }
            
            await self.ws.send(json.dumps(config))
            logger.info(f"📤 Sent Soniox config: model={self.model}, languages={self.language_hints}")
            
            # Wait briefly for acknowledgment or error
            try:
                first_response = await asyncio.wait_for(self.ws.recv(), timeout=2.0)
                response_data = json.loads(first_response)
                if "error" in response_data:
                    logger.error(f"❌ Soniox config rejected: {response_data['error']}")
                    raise Exception(f"Soniox config error: {response_data['error']}")
                else:
                    logger.info(f"✅ Soniox config accepted: {response_data}")
            except asyncio.TimeoutError:
                logger.warning("⚠️ No immediate response from Soniox (might be normal)")
            except json.JSONDecodeError:
                logger.warning("⚠️ Soniox sent non-JSON response (might be binary)")
            
            self.is_connected = True
            
            # Start receive loop
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.receive_task.set_name("Soniox_Receive_Loop")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Soniox: {e}", exc_info=True)
            raise
    
    async def send_audio(self, audio_chunk: bytes):
        """
        Send raw PCM audio bytes (binary frame).
        """
        if not self.is_connected or not self.ws:
            logger.warning("⚠️ Soniox not connected, skipping audio chunk")
            return
        
        try:
            await self.ws.send(audio_chunk)
        except Exception as e:
            logger.error(f"❌ Error sending audio to Soniox: {e}")
            self.is_connected = False
    
    async def _receive_loop(self):
        """
        Receive and parse Soniox responses.
        Accumulates tokens and emits final transcripts.
        """
        try:
            async for message in self.ws:
                try:
                    response = json.loads(message)
                    
                    # Check for errors
                    if "error" in response:
                        logger.error(f"❌ Soniox error: {response['error']}")
                        continue
                    
                    # Check for finish signal
                    if response.get("finished"):
                        logger.info("✅ Soniox stream finished")
                        break
                    
                    # Parse tokens
                    tokens = response.get("tokens", [])
                    if not tokens:
                        continue
                    
                    # Accumulate final tokens
                    final_tokens = [t for t in tokens if t.get("is_final")]
                    if final_tokens:
                        # Build transcript from final tokens
                        transcript_text = "".join([t.get("text", "") for t in final_tokens])
                        
                        if transcript_text.strip():
                            # Get timestamp from first token
                            timestamp_ms = final_tokens[0].get("start_ms", time.time() * 1000)
                            
                            logger.info(f"📝 Soniox final transcript: '{transcript_text}'")
                            
                            # Emit callback
                            if self.on_final_transcript:
                                try:
                                    if asyncio.iscoroutinefunction(self.on_final_transcript):
                                        await self.on_final_transcript(transcript_text, timestamp_ms)
                                    else:
                                        self.on_final_transcript(transcript_text, timestamp_ms)
                                except Exception as e:
                                    logger.error(f"❌ Error in Soniox callback: {e}", exc_info=True)
                
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse Soniox response: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing Soniox message: {e}", exc_info=True)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Soniox connection closed")
        except Exception as e:
            logger.error(f"❌ Soniox receive loop error: {e}", exc_info=True)
        finally:
            self.is_connected = False
    
    async def close(self):
        """
        Gracefully close the WebSocket connection.
        Sends empty frame and waits for 'finished' signal.
        """
        if not self.ws:
            return
        
        try:
            logger.info("🛑 Closing Soniox connection...")
            
            # Send empty frame to signal end
            await self.ws.send(b"")
            
            # Wait briefly for 'finished' response (handled in receive_loop)
            await asyncio.sleep(0.2)
            
            # Cancel receive task
            if self.receive_task and not self.receive_task.done():
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass
            
            # Close socket
            await self.ws.close()
            logger.info("✅ Soniox connection closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing Soniox: {e}", exc_info=True)
        finally:
            self.is_connected = False
            self.ws = None
