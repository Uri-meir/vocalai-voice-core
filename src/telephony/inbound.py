import logging
from fastapi import APIRouter, Form, Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from src.config.environment import config
from src.config.supabase_client import get_supabase_client

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/inbound")
async def twilio_inbound(
    To: str = Form(...),
    From: str = Form(...),
    CallSid: str = Form(...),
):
    """
    Handle inbound Twilio calls. 
    Lookup assistant by 'To' number.
    'From' is the customer number.
    """
    try:
        logger.info(f"📞 Inbound Call: To={To}, CallSid={CallSid}")
        
        supabase = get_supabase_client()
        
        # 1. Resolve Assistant
        # Find the first assistant created with this phone number that has an internal ID
        assistants_resp = supabase.table("assistants")\
            .select("professional_slug, internal_assistant_id")\
            .eq("phone_number_e164", To)\
            .not_.is_("internal_assistant_id", "null")\
            .order("created_at")\
            .limit(1)\
            .execute()
            
        if not assistants_resp.data:
            logger.warning(f"⚠️ No configured assistant found for number: {To}")
            resp = VoiceResponse()
            resp.say("Sorry, this number is not configured in the system.", language="en-US")
            return Response(content=str(resp), media_type="text/xml")
            
        row = assistants_resp.data[0]
        professional_slug = row["professional_slug"]
        internal_assistant_id = row["internal_assistant_id"]
        
        # 2. Validate Config Exists
        # Verify the internal ID actually points to a valid config
        config_resp = supabase.table("voice_assistant_configs")\
            .select("id")\
            .eq("id", internal_assistant_id)\
            .execute()
            
        if not config_resp.data:
             logger.error(f"❌ Assistant Config matches but row missing for ID: {internal_assistant_id}")
             resp = VoiceResponse()
             resp.say("Sorry, technical configuration error.", language="en-US")
             return Response(content=str(resp), media_type="text/xml")

        # 3. Build TwiML with Media Stream
        resp = VoiceResponse()
        connect = Connect()
        
        public_url = config.get("twilio.public_url").strip() if config.get("twilio.public_url") else ""
        if not public_url:
             logger.error("❌ PUBLIC_URL not set in config")
             resp = VoiceResponse()
             resp.say("System configuration error.")
             return Response(content=str(resp), media_type="text/xml")
             
        # Construct WSS URL
        # We need to construct the stream URL manually to include query params for the websocket connection
        # Expected WS Path: /twilio/media-stream
        stream_base = public_url.replace("https://", "wss://").replace("http://", "ws://")
        stream_url = f"{stream_base}/twilio/media-stream"

        # Create Stream Element
        stream = connect.stream(url=stream_url)
        
        # Add Parameters (Sent to the Server upon connection)
        stream.parameter(name="assistant_id", value=internal_assistant_id) # Using 'assistant_id' key to match existing media_stream.py logic
        stream.parameter(name="professional_slug", value=professional_slug)
        stream.parameter(name="direction", value="inbound")
        stream.parameter(name="customer_number", value=From) # The caller is the customer
        # Wait, the request said: "Twilio will POST ... To -> the called number"
        # Usually for INBOUND, 'Caller' is the customer, 'To' is the system.
        # But 'voice_hook.py' used 'customer_number' param.
        # In 'media_stream.py', it looks for 'customer_number' in custom params.
        # If this is INBOUND, the customer is the CALLER.
        # BUT the logic asks to look up assistant by 'To'.
        # I should probably pass 'From' as customer number if available, but the prompt spec 
        # only explicitly asked for To and CallSid in the function signature.
        # I will add 'From' to the signature to be safe and accurate.
        
        resp.append(connect)
        
        logger.info(f"✅ Connecting call {CallSid} to assistant {internal_assistant_id}")
        return Response(content=str(resp), media_type="text/xml")

    except Exception as e:
        logger.error(f"❌ Error handling inbound call: {e}")
        # Fail safe TwiML
        resp = VoiceResponse()
        resp.say("Sorry, something went wrong.")
        return Response(content=str(resp), media_type="text/xml")
