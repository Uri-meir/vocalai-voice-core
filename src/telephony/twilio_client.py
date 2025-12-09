import logging
from twilio.rest import Client
from src.config.environment import config

logger = logging.getLogger(__name__)

class TwilioClientWrapper:
    def __init__(self):
        self.client = Client(config.get("twilio.account_sid"), config.get("twilio.auth_token"))
        self.phone_number = config.get("twilio.phone_number")
        self.public_url = config.get("twilio.public_url")

    def make_call(self, to: str):
        """Initiates an outbound call to the specified number."""
        url = f"{self.public_url}/twilio/voice-hook"
        try:
            call = self.client.calls.create(
                to=to,
                from_=self.phone_number,
                url=url
            )
            logger.info(f"📞 Call initiated: {call.sid}")
            return call.sid
        except Exception as e:
            logger.error(f"❌ Failed to make call: {e}")
            raise
