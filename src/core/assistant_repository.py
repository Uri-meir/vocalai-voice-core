from abc import ABC, abstractmethod
from typing import Optional
from src.core.assistant_config import AssistantConfig

class AssistantRepository(ABC):
    """
    Interface for storing and retrieving Assistant Configurations.
    """
    @abstractmethod
    async def get_by_id(self, assistant_id: str) -> Optional[AssistantConfig]:
        """Retrieve an assistant configuration by its ID."""
        pass

    @abstractmethod
    async def upsert(self, config: AssistantConfig) -> AssistantConfig:
        """Create or update an assistant configuration."""
        pass
