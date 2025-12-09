from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import json
import base64
import asyncio
from src.telephony.audio_utils import mulaw_to_pcm, pcm_to_mulaw, resample_audio
from src.gemini.client import GeminiLiveClient
from src.config.environment import config

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio Media Stream WebSocket and Bridge to Gemini."""
    await websocket.accept()
    logger.info("🔌 Twilio Media Stream Connected")
    
    # queues for bridging
    mic_queue = asyncio.Queue()     # Twilio -> Gemini
    speaker_queue = asyncio.Queue() # Gemini -> Twilio
    
    # Initialize Gemini Client
    client = GeminiLiveClient(input_queue=mic_queue, output_queue=speaker_queue)
    
    stream_sid = None
    gemini_task = None
    sender_task = None
    
    try:
        # Start Gemini Session
        system_instruction = config.get_system_instruction() or "You are a helpful assistant."
        gemini_task = asyncio.create_task(client.start(system_instruction=system_instruction))
        
        # Trigger Initial Greeting (if configured)
        greeting_text = config.get("gemini.greeting_text")
        if greeting_text:
            # Wait briefly for connection (client.start is async but we need session ready)
            # A better way is to wait for a signal, but for POC we'll wait a startup delay or modify client.start to wait
            # Here we just execute it as a task that retries or waits
            async def trigger_greeting():
                # Wait for session to be established
                for _ in range(10): 
                    if client.session:
                        break
                    await asyncio.sleep(0.5)
                
                if client.session:
                    # Instruct the model to speak the greeting
                    await client.send_text(f"The user has joined. Say exactly this to start the conversation: '{greeting_text}'")
            
            asyncio.create_task(trigger_greeting())
        
        # Start Outbound Sender (Gemini -> Twilio)
        async def send_audio_to_twilio():
            while True:
                try:
                    # Get PCM24k/16k from Gemini
                    chunk = await speaker_queue.get()
                    if not chunk: continue
                    
                    # Resample to 8kHz for Twilio
                    # Assuming Gemini sends 24kHz (RECEIVE_SAMPLE_RATE)
                    # NOTE: Gemini might default to 24k. We need to match config.
                    in_rate = config.get("audio.receive_sample_rate", 24000)
                    resampled = resample_audio(chunk, in_rate, 8000)
                    
                    # Encode to mulaw
                    payload = base64.b64encode(pcm_to_mulaw(resampled)).decode("utf-8")
                    
                    # Send to Twilio
                    if stream_sid:
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload}
                        })
                except Exception as e:
                    logger.error(f"Error sending to Twilio: {e}")
                    break

        sender_task = asyncio.create_task(send_audio_to_twilio())

        # Inbound Loop (Twilio -> Gemini)
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")
            
            if event == "connected":
                logger.info(f"✅ Twilio Connected: {data}")
                
            elif event == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                logger.info(f"🏁 Stream Started: {stream_sid}")
                
            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    # Decode Base64 -> mulaw -> PCM16 (8kHz)
                    pcm_8k = mulaw_to_pcm(base64.b64decode(payload))
                    
                    # Resample 8k -> 16k (SEND_SAMPLE_RATE)
                    out_rate = config.get("audio.send_sample_rate", 16000)
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
        logger.error(f"❌ Media Stream Error: {e}")
    finally:
        # Cleanup
        if gemini_task: gemini_task.cancel()
        if sender_task: sender_task.cancel()
        try:
            await websocket.close()
        except:
            pass
        logger.info("👋 Media Stream Cleanup Complete")
