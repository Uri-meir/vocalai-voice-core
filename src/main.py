import logging
import uvicorn
from fastapi import FastAPI
from src.config.environment import config
from src.utils.logging_setup import setup_logging
from src.telephony.voice_hook import router as voice_router
from src.telephony.media_stream import router as media_router
from src.telephony.twilio_client import TwilioClientWrapper
from fastapi import Body, HTTPException
from src.api.models import StartCallRequest, StartCallResponse

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Gemini Telephony Server")

# Include Routers
app.include_router(voice_router, prefix="/twilio")
app.include_router(media_router, prefix="/twilio")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Gemini Telephony Server...")
    config.validate()
    logger.info(f"📞 Twilio Number: {config.get('twilio.phone_number')}")
    logger.info(f"🌍 Public URL: {config.get('twilio.public_url')}")

@app.post("/call/start", response_model=StartCallResponse)
async def start_call(request: StartCallRequest):
    """Initiate an outbound call."""
    logger.info(f"📤 Starting outbound call to {request.customer.number} (assistant_id={request.assistantId})")
    
    try:
        client = TwilioClientWrapper()
        # Note: phoneNumberId is received but not yet resolved to a real phone number in this POC.
        # We continue to use the default configured number.
        call = client.make_call(
            to=request.customer.number, 
            assistant_id=request.assistantId
        )
        
        return StartCallResponse(
            success=True,
            callId=call.sid,
            status=call.status or "queued",
            message="Call started successfully"
        )
    except Exception as e:
        logger.error(f"Failed to start call: {e}")
        # Return HTTP 500 with error details as requested
        # Or return a 200 with success=False if preferred, but user said "return an HTTP 500"
        # However, to return the StartCallResponse model structure inside 500, we might need JSONResponse
        # For typical FastAPI, raising HTTPException(500, detail=...) is standard but returns a generic detail dict.
        # User requested: "Return an HTTP 500 with a StartCallResponse where success=False..."
        # This requires a custom JSONResponse or handling expected errors without raising HTTPException.
        # Let's try to return the JSON with status_code=500.
        
        from fastapi import status
        from fastapi.responses import JSONResponse
        from fastapi.encoders import jsonable_encoder
        
        error_response = StartCallResponse(
            success=False,
            callId="",
            status="error",
            message=str(e)
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder(error_response)
        )

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
