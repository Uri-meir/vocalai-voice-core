from fastapi import APIRouter, Request, Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from src.config.environment import config
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/voice-hook")
async def voice_hook(request: Request):
    """Handle incoming calls and start a Media Stream."""
    logger.info("📞 Incoming call received")
    
    resp = VoiceResponse()
    # Removed static greeting to allow AI to speak first
    # resp.say("Connecting you to your AI assistant.")
    
    connect = Connect()
    public_url = config.get("twilio.public_url")
    if not public_url:
        logger.error("❌ PUBLIC_URL not set in config")
        resp.say("System configuration error.")
        return Response(content=str(resp), media_type="application/xml")

    # Clean the URL (remove https://)
    stream_url = public_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url += "/twilio/media-stream"
    
    connect.stream(url=stream_url)
    resp.append(connect)
    
    return Response(content=str(resp), media_type="application/xml")
