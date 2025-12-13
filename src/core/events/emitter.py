import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from functools import lru_cache

import httpx
from src.config.environment import config as env_config

logger = logging.getLogger(__name__)

class SupabaseVapiWebhookEmitter:
    def __init__(self, webhook_url: Optional[str] = None, timeout_seconds: float = 5.0) -> None:
        self._webhook_url = webhook_url or os.getenv("SUPABASE_VAPI_WEBHOOK_URL")
        if not self._webhook_url:
            # Fallback to config if not in os.environ directly (depending on how config is loaded)
            self._webhook_url = env_config.get("supabase.vapi_webhook_url")
        
        if not self._webhook_url:
             logger.warning("⚠️ SUPABASE_VAPI_WEBHOOK_URL is not set. Webhooks will not be sent.")
        
        self._timeout_seconds = timeout_seconds

    async def _emit(self, payload: Dict[str, Any], event_type: str, call_id: str) -> None:
        if not self._webhook_url:
            return

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            try:
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                logger.info(f"✅ Emitted {event_type} to Supabase webhook for call_id={call_id}")
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ Failed to emit {event_type}: HTTP {e.response.status_code} - {e.response.text}")
            except Exception as e:
                logger.error(f"❌ Failed to emit {event_type}: {e}")

    async def emit_call_started(
        self,
        call_id: str,
        assistant_id: str,
        customer_number: str,
        created_at: Optional[datetime] = None,
    ) -> None:
        if not created_at:
            created_at = datetime.now(timezone.utc)
        
        payload = {
            "message": {
                "type": "call.started",
                "call": {
                    "id": call_id,
                    "assistantId": assistant_id,
                    "customer": {
                        "number": customer_number
                    },
                    "createdAt": created_at.isoformat().replace("+00:00", "Z")
                }
            }
        }
        await self._emit(payload, "call.started", call_id)

    async def emit_meeting_scheduled(
        self,
        call_id: str,
        assistant_id: str,
        meeting_details: Dict[str, Any]
    ) -> None:
        payload = {
            "message": {
                "type": "meeting.scheduled",
                "call": {
                    "id": call_id,
                    "assistantId": assistant_id
                },
                "meeting": meeting_details
            }
        }
        await self._emit(payload, "meeting.scheduled", call_id)

    async def emit_call_ended(
        self,
        call_id: str,
        assistant_id: str,
        customer_number: str,
        created_at: datetime,
        ended_at: datetime,
        transcript: Optional[str] = None,
        ended_reason: Optional[str] = "completed",
        minutes: Optional[float] = None,
    ) -> None:
        
        if minutes is None:
            # Calculate duration in minutes
            diff = (ended_at - created_at).total_seconds()
            minutes = diff / 60.0
            
        payload = {
            "message": {
                "type": "call.ended",
                "call": {
                    "id": call_id,
                    "assistantId": assistant_id,
                    "customer": {
                        "number": customer_number
                    },
                    "createdAt": created_at.isoformat().replace("+00:00", "Z"),
                    "endedAt": ended_at.isoformat().replace("+00:00", "Z"),
                    "transcript": transcript or "",
                    "endedReason": ended_reason
                },
                "costs": [
                    {
                        "type": "vapi",
                        "minutes": minutes
                    }
                ]
            }
        }
        await self._emit(payload, "call.ended", call_id)

@lru_cache(maxsize=1)
def get_supabase_vapi_webhook_emitter() -> SupabaseVapiWebhookEmitter:
    return SupabaseVapiWebhookEmitter()
