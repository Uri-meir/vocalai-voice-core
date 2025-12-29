import logging
import os
import uvicorn
from fastapi import FastAPI
from src.config.environment import config
from src.utils.logging_setup import setup_logging
from src.telephony.voice_hook import router as voice_router
from src.telephony.inbound import router as inbound_router
from src.telephony.media_stream import router as media_router
from src.api.routes.assistants import router as assistants_router
from src.api.routes.phone_numbers import router as phone_numbers_router
from src.telephony.twilio_client import TwilioClientWrapper
from fastapi import Body

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Gemini Telephony Server")

# Setup CORS (Critical for WebSockets behind App Runner LB)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(voice_router, prefix="/twilio")
app.include_router(inbound_router, prefix="/twilio")
app.include_router(media_router, prefix="/twilio")
app.include_router(assistants_router)
app.include_router(phone_numbers_router)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Gemini Telephony Server...")
    config.validate()
    logger.info(f"📞 Twilio Number: {config.get('twilio.phone_number')}")
    logger.info(f"🌍 Public URL: {config.get('twilio.public_url')}")

from fastapi import HTTPException
from src.api.models import StartCallRequest
from src.core.assistants_repository_factory import get_assistant_repository

@app.get("/debug-logs")
async def debug_logs():
    """Endpoint to verify deployment and test logging."""
    import time
    ts = time.time()
    msg = f"🔍 DEBUG LOG TEST - Timestamp: {ts}"
    print(f"PRINT: {msg}", flush=True)
    logger.info(f"LOGGER: {msg}")
    return {"message": "Logged successfully", "timestamp": ts, "version": "v1-unbuffered-check"}

@app.post("/call/start")
async def start_call(payload: StartCallRequest):
    """Initiate an outbound call."""
    logger.info(f"📤 Starting outbound call to {payload.to} with assistant {payload.assistant_id}")
    
    # Validate Assistant exists
    repo = get_assistant_repository()
    config = await repo.get_by_id(payload.assistant_id)
    if not config:
         logger.error(f"❌ Assistant {payload.assistant_id} not found")
         raise HTTPException(status_code=404, detail="Assistant configuration not found")

    # Resolve Outbound Phone Number (Caller ID)
    # We use the explicit number from payload if provided (dynamic caller ID)
    from_number = payload.twilioPhoneNumber
    if from_number:
        logger.info(f"📞 Using Dynamic Caller ID from payload: {from_number}")

    client = TwilioClientWrapper()
    sid = client.make_call(
        to=payload.to, 
        assistant_id=payload.assistant_id, 
        customer_number=payload.to,
        from_number=from_number
    )
    return {"call_sid": sid, "status": "initiated", "assistant_id": payload.assistant_id}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
