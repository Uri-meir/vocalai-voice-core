from typing import Optional
from uuid import uuid4
from src.core.assistant_config import AssistantConfig
from src.core.assistant_repository import AssistantRepository
from src.config.supabase_client import get_supabase_client

import logging

logger = logging.getLogger(__name__)

class SupabaseAssistantRepository(AssistantRepository):
    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = "voice_assistant_configs"

    async def get_by_id(self, assistant_id: str) -> Optional[AssistantConfig]:
        try:
            response = self._client.table(self._table).select("*").eq("id", assistant_id).single().execute()
            if not response.data:
                return None
            
            row = response.data
            return AssistantConfig(
                id=row["id"],
                professional_slug=row["professional_slug"],
                display_name=row["display_name"],
                language=row["language"],
                system_prompt=row["system_prompt"],
                first_message=row.get("first_message"),
                llm_model=row["llm_model"],
                tts_model=row.get("tts_model"),
                voice_id=row.get("voice_id"),
                timezone=row["timezone"],
                silence_timeout_seconds=row["silence_timeout_seconds"],
                background_denoising_enabled=row["background_denoising_enabled"],
                metadata=row.get("metadata"),
            )
        except Exception as e:
            logger.error(f"Error fetching assistant {assistant_id}: {e}")
            return None

    async def upsert(self, config: AssistantConfig) -> AssistantConfig:
        if config.id is None:
            config.id = uuid4().hex

        row = {
            "id": config.id,
            "professional_slug": config.professional_slug,
            "display_name": config.display_name,
            "language": config.language,
            "system_prompt": config.system_prompt,
            "first_message": config.first_message,
            "llm_model": config.llm_model,
            "tts_model": config.tts_model,
            "voice_id": config.voice_id,
            "timezone": config.timezone,
            "silence_timeout_seconds": config.silence_timeout_seconds,
            "background_denoising_enabled": config.background_denoising_enabled,
            "metadata": config.metadata,
        }

        try:
            self._client.table(self._table).upsert(row, on_conflict="id").execute()
            return config
        except Exception as e:
            logger.error(f"Error upserting assistant: {e}")
            raise e
