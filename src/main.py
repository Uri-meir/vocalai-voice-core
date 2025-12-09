import logging
import uvicorn
from fastapi import FastAPI
from src.config.environment import config
from src.utils.logging_setup import setup_logging
from src.telephony.voice_hook import router as voice_router
from src.telephony.media_stream import router as media_router
from src.telephony.twilio_client import TwilioClientWrapper
from fastapi import Body

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

@app.post("/call/start")
async def start_call(to: str = Body(..., embed=True)):
    """Initiate an outbound call."""
    logger.info(f"📤 Starting outbound call to {to}")
    client = TwilioClientWrapper()
    sid = client.make_call(to)
    return {"call_sid": sid, "status": "initiated"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
