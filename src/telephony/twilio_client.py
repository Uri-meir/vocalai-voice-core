import logging
from urllib.parse import quote
from twilio.rest import Client
from src.config.environment import config

logger = logging.getLogger(__name__)

class TwilioClientWrapper:
    def __init__(self):
        self.client = Client(config.get("twilio.account_sid"), config.get("twilio.auth_token"))
        self.phone_number = config.get("twilio.phone_number")
        self.public_url = config.get("twilio.public_url")

    def make_call(self, to: str, from_number: str = None, assistant_id: str = None):
        """Initiates an outbound call to the specified number."""
        url = f"{self.public_url}/twilio/voice-hook"
        # Pass parameters to webhook
        params = []
        if assistant_id:
            params.append(f"assistant_id={quote(assistant_id)}")
        params.append(f"customer_number={quote(to)}")
        
        if params:
            url += "?" + "&".join(params)
        # Use provided from_number or fallback to config default
        final_from = from_number or self.phone_number
        
        try:
            call = self.client.calls.create(
                to=to,
                from_=final_from,
                url=url
            )
            logger.info(f"📞 Call initiated: {call.sid}")
            return call
        except Exception as e:
            logger.error(f"❌ Failed to make call: {e}")
            raise
