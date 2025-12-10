from typing import Dict, Optional
from src.core.session import CallSession

# CallSid -> CallSession
sessions: Dict[str, CallSession] = {}

def get_session(call_id: str) -> Optional[CallSession]:
    return sessions.get(call_id)

def remove_session(call_id: str) -> None:
    if call_id in sessions:
        del sessions[call_id]
