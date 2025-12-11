import logging
from twilio.rest import Client
from src.config.environment import config

logger = logging.getLogger(__name__)

class TwilioClientWrapper:
    def __init__(self):
        self.client = Client(config.get("twilio.account_sid"), config.get("twilio.auth_token"))
        self.phone_number = config.get("twilio.phone_number")
        self.public_url = config.get("twilio.public_url")

    def make_call(self, to: str, assistant_id: str, customer_number: str = None):
        """Initiates an outbound call to the specified number."""
        # We pass customer_number in query to forward to media stream for logging
        url = f"{self.public_url}/twilio/voice-hook?assistant_id={assistant_id}&customer_number={customer_number or to}"
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
