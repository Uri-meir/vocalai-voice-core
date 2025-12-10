from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.core.events import EventEmitter

@dataclass
class CallSessionConfig:
    call_id: str
    assistant_id: str
    customer_number: str

class CallSession:
    def __init__(self, cfg: CallSessionConfig, emitter: EventEmitter):
        self.cfg = cfg
        self.emitter = emitter
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None

    async def mark_started(self) -> None:
        """Marks the session as started and emits the event."""
        self.started_at = datetime.utcnow()
        await self.emitter.emit_call_started(
            call_id=self.cfg.call_id,
            assistant_id=self.cfg.assistant_id,
            customer_number=self.cfg.customer_number,
            created_at=self.started_at
        )

    async def mark_ended(self, transcript: Optional[str], minutes: float) -> None:
        """Marks the session as ended and emits the event."""
        self.ended_at = datetime.utcnow()
        # Fallback if started_at is somehow None (e.g. restart)
        start_time = self.started_at or self.ended_at
        
        await self.emitter.emit_call_ended(
            call_id=self.cfg.call_id,
            assistant_id=self.cfg.assistant_id,
            created_at=start_time,
            ended_at=self.ended_at,
            minutes=minutes,
            transcript=transcript
        )
