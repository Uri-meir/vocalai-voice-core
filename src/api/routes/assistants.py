from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from src.core.assistant_config import AssistantConfig
from src.core.assistant_repository import AssistantRepository
from src.core.assistants_repository_factory import get_assistant_repository

router = APIRouter(prefix="/assistants", tags=["assistants"])

def get_repo() -> AssistantRepository:
    return get_assistant_repository()

# Response Model
class AssistantResponse(BaseModel):
    id: str
    professional_slug: str
    display_name: str
    language: str
    system_prompt: str
    first_message: Optional[str]
    llm_model: str
    tts_model: Optional[str]
    voice_id: Optional[str]
    timezone: Optional[str]
    silence_timeout_seconds: int
    background_denoising_enabled: bool
    metadata: Optional[Dict[str, Any]]

    @classmethod
    def from_config(cls, config: AssistantConfig) -> "AssistantResponse":
        return cls(
            id=config.id,
            professional_slug=config.professional_slug,
            display_name=config.display_name,
            language=config.language,
            system_prompt=config.system_prompt,
            first_message=config.first_message,
            llm_model=config.llm_model,
            tts_model=config.tts_model,
            voice_id=config.voice_id,
            timezone=config.timezone,
            silence_timeout_seconds=config.silence_timeout_seconds,
            background_denoising_enabled=config.background_denoising_enabled,
            metadata=config.metadata
        )

@router.post("/", response_model=AssistantResponse)
async def create_assistant(
    payload: Dict[str, Any],
    repo: AssistantRepository = Depends(get_repo)
):
    """
    Create or update an assistant from a Vapi-style payload (N8N).
    """
    # Map using Pydantic factory
    config = AssistantConfig.from_vapi_payload(payload)
    
    # Persist
    saved_config = await repo.upsert(config)
    
    # Return
    return AssistantResponse.from_config(saved_config)
