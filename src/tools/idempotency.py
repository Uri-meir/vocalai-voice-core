import hashlib
import logging
from typing import Dict, Optional, Any
from src.tools.schemas import ToolResult

logger = logging.getLogger(__name__)

class IdempotencyLayer:
    """
    Prevents double-execution of side-effect tools (like booking).
    Currently uses in-memory cache.
    """
    def __init__(self):
        self._cache: Dict[str, ToolResult] = {}

    def _compute_key(self, call_id: str, tool_name: str, args: Dict[str, Any]) -> str:
        # Create a stable string representation
        # Sort keys to ensure consistent order
        args_str = str(sorted(args.items()))
        raw = f"{call_id}:{tool_name}:{args_str}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_cached_result(self, call_id: str, tool_name: str, args: Dict[str, Any]) -> Optional[ToolResult]:
        key = self._compute_key(call_id, tool_name, args)
        result = self._cache.get(key)
        if result:
            logger.info(f"🔄 Returning cached result for {tool_name} (key={key[:8]})")
        return result

    def cache_result(self, call_id: str, tool_name: str, args: Dict[str, Any], result: ToolResult):
        if result.success:
            key = self._compute_key(call_id, tool_name, args)
            self._cache[key] = result
            logger.info(f"💾 Cached success result for {tool_name} (key={key[:8]})")
