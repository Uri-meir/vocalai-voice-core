from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

class AssistantConfig(BaseModel):
    """
    Configuration for a voice assistant.
    """
    id: Optional[str] = None  # Assigned by service
    professional_slug: str
    display_name: str
    language: str
    system_prompt: str
    llm_model: str
    voice_id: Optional[str] = None
    tts_model: Optional[str] = None
    first_message: Optional[str] = None
    timezone: Optional[str] = None
    silence_timeout_seconds: int = 60
    background_denoising_enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_vapi_payload(cls, payload: Dict[str, Any]) -> "AssistantConfig":
        """
        Create an AssistantConfig from a Vapi-style provisioning payload.
        """
        metadata = payload.get("metadata", {}) or {}
        model_config = payload.get("model", {}) or {}
        voice_config = payload.get("voice", {}) or {}
        transcriber_config = payload.get("transcriber", {}) or {}

        # Extract system prompt
        system_prompt = ""
        messages = model_config.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                    break

        return cls(
            professional_slug=metadata.get("professional_slug", ""),
            display_name=payload.get("name", "Voice Agent"),
            language=transcriber_config.get("language", "he"),
            system_prompt=system_prompt,
            llm_model=model_config.get("model", ""),
            voice_id=voice_config.get("voiceId"),
            tts_model=voice_config.get("model"),
            first_message=payload.get("firstMessage"),
            timezone=metadata.get("timezone", "Asia/Jerusalem"),
            silence_timeout_seconds=payload.get("silenceTimeoutSeconds", 60),
            background_denoising_enabled=payload.get("backgroundDenoisingEnabled", True),
            metadata=metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dict representation for DB persistence.
        """
        return self.model_dump()

        return self.model_dump()
