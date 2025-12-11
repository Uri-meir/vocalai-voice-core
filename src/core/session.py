from dataclasses import dataclass
from src.core.assistant_config import AssistantConfig

@dataclass
class CallSession:
    call_id: str
    assistant_config: AssistantConfig
    stream_sid: str = None
