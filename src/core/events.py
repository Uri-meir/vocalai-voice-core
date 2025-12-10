import httpx
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.config.environment import config

logger = logging.getLogger(__name__)

# Global reused client
_http_client: Optional[httpx.AsyncClient] = None

async def close_event_client():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None

class EventEmitter:
    def __init__(self):
        self.webhook_url = config.get("supabase.vapi_webhook_url")
        if not self.webhook_url:
            logger.warning("⚠️ SUPABASE_VAPI_WEBHOOK_URL not set. Events will not be sent.")

    async def _post(self, payload: Dict[str, Any]):
        """Helper to send POST request using shared client."""
        if not self.webhook_url:
            return

        # Use the module-level client
        global _http_client
        if _http_client is None:
            _http_client = httpx.AsyncClient(timeout=10.0)

        try:
            response = await _http_client.post(self.webhook_url, json=payload)
            # response.raise_for_status() 
            if response.status_code >= 400:
                 logger.error(f"❌ Failed to send event (HTTP {response.status_code}): {response.text}")
            else:
                 logger.info(f"✅ Event sent to Supabase: {payload.get('message', {}).get('type')}")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP Error: {e}")
        except Exception as e:
            logger.error(f"❌ Connection Error: {e}")

    async def emit_call_started(self, call_id: str, assistant_id: str, customer_number: str, created_at: datetime):
        """Emits call.started event."""
        payload = {
            "message": {
                "type": "call.started",
                "call": {
                    "id": call_id,
                    "assistantId": assistant_id,
                    "customer": {"number": customer_number},
                    "createdAt": created_at.isoformat()
                },
                "assistant": {"id": assistant_id}
            }
        }
        await self._post(payload)

    async def emit_call_ended(
        self, 
        call_id: str, 
        assistant_id: str, 
        created_at: datetime, 
        ended_at: datetime, 
        minutes: float, 
        transcript: Optional[str] = ""
    ):
        """Emits call.ended event."""
        payload = {
            "message": {
                "type": "call.ended",
                "call": {
                    "id": call_id,
                    "assistantId": assistant_id,
                    "createdAt": created_at.isoformat(),
                    "endedAt": ended_at.isoformat()
                },
                "assistant": {"id": assistant_id},
                "costs": [
                    {"type": "vapi", "minutes": minutes}
                ],
                "transcript": transcript or ""
            }
        }
        await self._post(payload)
