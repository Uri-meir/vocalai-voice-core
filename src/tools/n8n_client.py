import httpx
import logging
from typing import Any, Dict, Optional
from src.tools.schemas import ToolResult

logger = logging.getLogger(__name__)

# Hardcoded for now per instructions, ideally moved to config
N8N_WEBHOOK_URL = "https://aipro.app.n8n.cloud/webhook/57d6ee51-198b-485a-af5f-63cf7242c5f4"

class N8NClient:
    """
    Transport layer for executing tools via n8n.
    """
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            # "Authorization": "Bearer ... " # TODO: Add if user provides key
        }

    async def execute_tool(
        self, 
        tool_name: str, 
        args: Dict[str, Any], 
        context_data: Dict[str, Any]
    ) -> ToolResult:
        """
        Sends a tool execution request to n8n.
        Payload structure: { "tool": "name", "args": {...}, "call_id": "...", ... }
        """
        payload = {
            "tool": tool_name,
            "args": args,
            **context_data
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    N8N_WEBHOOK_URL,
                    json=payload,
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    # Handle empty body (n8n might return 200 OK with no content)
                    if not response.content:
                        logger.info(f"n8n returned 200 OK with empty body for {tool_name}")
                        return ToolResult.success_result(data={"status": "executed", "details": "No content returned from n8n"})
                    
                    try:
                        data = response.json()
                        return ToolResult.success_result(data=data)
                    except ValueError:
                        logger.warning(f"n8n returned 200 OK but invalid JSON: {response.text}")
                        # Fallback for plain text success
                        return ToolResult.success_result(data={"text_response": response.text})
                
                else:
                    logger.error(f"n8n returned status {response.status_code}: {response.text}")
                    # Allow retry on 5xx
                    retryable = response.status_code >= 500
                    return ToolResult.error_result(
                        "UPSTREAM_ERROR", 
                        f"n8n returned {response.status_code}",
                        retryable=retryable
                    )
                    
        except httpx.TimeoutException:
            logger.error(f"n8n request timed out for {tool_name}")
            return ToolResult.error_result("TIMEOUT", "Request to n8n timed out", retryable=True)
            
        except Exception as e:
            logger.exception(f"n8n client error for {tool_name}")
            return ToolResult.error_result("TransportError", str(e), retryable=False)
