import uuid
from typing import Optional, Dict
from src.core.assistant_config import AssistantConfig
from src.core.assistant_repository import AssistantRepository

class InMemoryAssistantRepository(AssistantRepository):
    """
    In-memory implementation of AssistantRepository.
    Stores assistants in a local Python dictionary.
    """
    def __init__(self):
        self._store: Dict[str, AssistantConfig] = {}

    async def get_by_id(self, assistant_id: str) -> Optional[AssistantConfig]:
        return self._store.get(assistant_id)

    async def upsert(self, config: AssistantConfig) -> AssistantConfig:
        if not config.id:
            config.id = uuid.uuid4().hex
        
        self._store[config.id] = config
        return config
