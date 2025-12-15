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
    connect = Connect()
    public_url = config.get("twilio.public_url")
    if not public_url:
        logger.error("❌ PUBLIC_URL not set in config")
        resp.say("System configuration error.")
        return Response(content=str(resp), media_type="application/xml")

    stream_url = public_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url += "/twilio/media-stream"
    
    # Get assistant_id and customer_number
    # 1. Try query params (for manual overrides or custom setups)
    assistant_id = request.query_params.get("assistant_id")
    customer_number = request.query_params.get("customer_number")

    # 2. Fallback: Try Twilio POST body (standard webhook behavior)
    if not customer_number:
        try:
            form_data = await request.form()
            customer_number = form_data.get("From")
            if not assistant_id:
                # Optional: Read assistant_id from body if somehow passed there, unlikely but possible
                pass
        except Exception:
            logger.warning("Could not parse form data")

    logger.info(f"🔍 Extracted: assistant_id={assistant_id}, customer_number={customer_number}")

    stream = connect.stream(url=stream_url)
    stream.parameter(name="assistant_id", value=assistant_id)
    stream.parameter(name="customer_number", value=customer_number)
    resp.append(connect)
    
    xml_content = str(resp)
    logger.info(f"📤 Generated TwiML: {xml_content}")
    
    return Response(content=xml_content, media_type="application/xml")
