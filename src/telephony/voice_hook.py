from fastapi import APIRouter, Request, Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Start
from twilio.request_validator import RequestValidator
from src.config.environment import config
import logging
import os
import httpx
import tempfile
from datetime import datetime, timezone
from src.config.supabase_client import get_supabase_client, upload_call_recording

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/voice-hook")
async def voice_hook(request: Request):
    """Handle incoming calls and start a Media Stream."""
    logger.info("📞 Incoming call received")
    
    public_url = config.get("twilio.public_url").strip() if config.get("twilio.public_url") else ""
    if not public_url:
        logger.error("❌ PUBLIC_URL not set in config")
        resp = VoiceResponse()
        resp.say("System configuration error.")
        return Response(content=str(resp), media_type="application/xml")
    
    resp = VoiceResponse()
    
    # Get assistant_id from query params
    assistant_id = request.query_params.get("assistant_id", "unknown")
    
    # Start recording using <Start><Recording> TwiML (works with Media Streams)
    # Pass assistant_id in callback URL so we can construct storage path later
    recording_callback_url = f"{public_url}/twilio/recording-callback?assistant_id={assistant_id}"
    start = Start()
    start.recording(recording_status_callback=recording_callback_url,
                   recording_status_callback_event='completed')
    resp.append(start)
    logger.info(f"📼 Recording enabled via <Start><Recording> TwiML (assistant: {assistant_id})")
    
    connect = Connect()

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

@router.post("/recording-callback")
async def recording_callback(request: Request):
    """
    Idempotent webhook called by Twilio when a recording is ready.
    Downloads recording to temp file, uploads to Supabase Storage, updates database.
    """
    temp_path = None
    try:
        # 1. Verify Twilio signature for security
        auth_token = config.get("twilio.auth_token")
        validator = RequestValidator(auth_token)
        
        # Get signature from headers
        signature = request.headers.get("X-Twilio-Signature", "")
        
        # Construct the correct URL for signature validation
        # In production with reverse proxy (Fly.io), need to check X-Forwarded-Proto
        url = str(request.url)
        
        # Fix: If behind reverse proxy, replace http:// with https://
        if request.headers.get("X-Forwarded-Proto") == "https" and url.startswith("http://"):
            url = url.replace("http://", "https://", 1)
            logger.debug(f"🔐 Adjusted URL for reverse proxy: {url}")
        
        # Get form data for validation
        form_data = await request.form()
        params = dict(form_data)
        
        # Debug logging for signature validation issues
        logger.info(f"🔐 Signature validation - URL: {url}")
        logger.debug(f"🔐 Signature: {signature[:20]}... | Params keys: {list(params.keys())}")
        
        # Verify signature
        if not validator.validate(url, params, signature):
            logger.error(f"❌ Invalid Twilio signature - URL: {url}")
            logger.error(f"❌ Check: 1) Correct auth_token 2) URL matches exactly (https/query params) 3) No proxy issues")
            return {"status": "error", "message": "Invalid signature"}, 403
        
        # 2. Extract Twilio params
        call_sid = form_data.get("CallSid")  # Use Twilio's name
        recording_sid = form_data.get("RecordingSid")
        recording_url = form_data.get("RecordingUrl")
        recording_duration = form_data.get("RecordingDuration")
        
        logger.info(
            f"📼 Recording ready: {recording_sid} for call {call_sid} "
            f"(duration: {recording_duration}s)"
        )
        
        # 3. Get assistant_id from query params (passed in recording callback URL)
        assistant_id = request.query_params.get("assistant_id", "unknown")
        
        # 4. Idempotency check
        supabase = get_supabase_client()
        call_log = supabase.table("call_logs")\
            .select("recording_url")\
            .eq("vapi_call_id", call_sid)\
            .execute()
        
        if call_log.data and call_log.data[0].get("recording_url"):
            logger.info(f"⏭️ Recording already processed for {call_sid}, skipping")
            return {"status": "already_processed", "call_sid": call_sid}
        
        # 5. Construct simple storage path (assistant_id/call_sid.mp3)
        # Both assistant_id and call_sid are UUIDs/SIDs, always URL-safe
        storage_path = f"{assistant_id}/{call_sid}.mp3"
        logger.info(f"📦 Storage path: {storage_path}")
        
        # Get customer number for metadata if call_logs row exists
        customer_number = call_log.data[0].get("user_id") if call_log.data else "unknown"
        
        # 7. Download to temp file (size-limited)
        account_sid = config.get("twilio.account_sid")
        download_url = f"{recording_url}.mp3" if not recording_url.endswith(".mp3") else recording_url
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "GET", download_url,
                    auth=(account_sid, auth_token),
                    timeout=120.0
                ) as response:
                    response.raise_for_status()
                    
                    # Stream with size limit
                    total_bytes = 0
                    max_bytes = 50 * 1024 * 1024  # 50MB
                    
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            raise ValueError(f"Recording exceeds 50MB limit")
                        temp_file.write(chunk)
        
        logger.info(f"⬇️ Downloaded: {total_bytes / (1024*1024):.2f} MB")
        
        # 8. Upload to Supabase
        result = upload_call_recording(temp_path, storage_path)
        
        logger.info(f"☁️ Uploaded to: {storage_path}")
        
        # 9. Create recording metadata
        recording_meta = {
            "assistant_id": assistant_id,
            "customer_number": customer_number,
            "bucket": "call-recordings",
            "path": storage_path,
            "recording_sid": recording_sid,
            "duration_s": int(recording_duration) if recording_duration else None,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        
        # 10. Update DB (only if call_logs row exists)
        if call_log.data:
            supabase.table("call_logs").update({
                "recording_url": result["signed_url"],
                "recording_meta": recording_meta
            }).eq("vapi_call_id", call_sid).execute()
            logger.info(f"💾 Updated call_logs for {call_sid}")
        else:
            logger.warning(f"⚠️ Call_logs row not found for {call_sid}, recording uploaded but not linked to DB")
        
        return {"status": "uploaded", "path": storage_path}
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error(f"❌ Twilio auth failed - check account_sid/auth_token")
        else:
            logger.error(f"❌ HTTP error downloading recording: {e}")
        return {"status": "error", "message": str(e)}
        
    except Exception as e:
        logger.error(f"❌ Recording callback error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
        
    finally:
        # Always clean up temp file
        if temp_path:
            try:
                os.unlink(temp_path)
                logger.debug(f"🗑️ Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up temp file: {e}")
