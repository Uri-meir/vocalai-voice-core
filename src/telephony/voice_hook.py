from fastapi import APIRouter, Request, Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from src.config.environment import config
import logging
from src.core.events import EventEmitter
from src.core.session import CallSession, CallSessionConfig
from src.core.session_store import sessions

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/voice-hook")
async def voice_hook(request: Request):
    """Handle incoming calls and start a Media Stream."""
    logger.info("📞 Incoming call received")
    
    # Parse form data
    form_data = await request.form()
    
    # robust retrieval of call_id (CallSid)
    import uuid
    call_sid = form_data.get("CallSid")
    if not call_sid:
        call_sid = str(uuid.uuid4())
        logger.warning(f"⚠️ No CallSid in request (Manual Test?). Generated temporary ID: {call_sid}")
        
    # Determine customer number:
    # 1. Query Param (Explicitly passed in outbound calls)
    # 2. 'From' field (Inbound calls)
    # 3. Fallback
    query_params = request.query_params
    assistant_id = query_params.get("assistant_id") or form_data.get("assistant_id") or "default-assistant"
    customer_number = query_params.get("customer_number") or form_data.get("From") or "unknown-number"

    # Initialize Session & Event Emitter
    try:
        emitter = EventEmitter()
        cfg = CallSessionConfig(
            call_id=call_sid,
            assistant_id=assistant_id,
            customer_number=customer_number
        )
        session = CallSession(cfg, emitter)
        
        # Store session
        sessions[call_sid] = session
        
        # Emit call.started
        await session.mark_started()
    except Exception as e:
        logger.error(f"❌ Error during session initialization: {e}")
        # Proceed anyway so the call connects!
    
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
