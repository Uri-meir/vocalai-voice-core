import os
from src.core.assistant_repository import AssistantRepository

_repo: AssistantRepository = None

def get_assistant_repository() -> AssistantRepository:
    global _repo
    if _repo is not None:
        return _repo

    backend = os.getenv("ASSISTANTS_REPO_BACKEND", "memory").lower()
    
    if backend == "supabase":
        from src.core.assistants_supabase import SupabaseAssistantRepository
        _repo = SupabaseAssistantRepository()
    else:
        from src.core.assistant_repository_inmemory import InMemoryAssistantRepository
        _repo = InMemoryAssistantRepository()
    
    return _repo
