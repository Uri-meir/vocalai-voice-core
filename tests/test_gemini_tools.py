import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.gemini.client import GeminiLiveClient
from src.tools.registry import ToolRegistry, ToolContext
from google.genai import types

@pytest.mark.asyncio
async def test_client_injects_tools():
    # Setup Registry
    registry = ToolRegistry()
    from pydantic import BaseModel
    class FakeArgs(BaseModel):
        x: int
    
    @registry.register("fake_tool", "description", FakeArgs)
    async def fake_executor(args, ctx):
        return "ok"

    # Setup Client
    client = GeminiLiveClient(
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        tool_registry=registry
    )

    # Mock genai.Client
    mock_files = MagicMock()
    client.client = mock_files
    
    # Mock the connect context manager
    mock_connect = AsyncMock()
    mock_session = AsyncMock()
    mock_connect.__aenter__.return_value = mock_session
    mock_files.aio.live.connect.return_value = mock_connect

    # Run start
    with patch("src.gemini.client.config") as mock_config:
        mock_config.GEMINI_API_KEY = "fake"
        mock_config.get.side_effect = lambda k, d=None: "model-id" if k == "gemini.model_id" else d
        
        # We start it as a task and cancel it immediately because it has infinite loops
        task = asyncio.create_task(client.start())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Verify config passed to connect
    args, kwargs = mock_files.aio.live.connect.call_args
    config_params = kwargs["config"]
    
    assert "tools" in config_params
    assert len(config_params["tools"]) == 1
    tool_obj = config_params["tools"][0]
    
    # Verify it is a Tool object
    assert isinstance(tool_obj, types.Tool)
    assert len(tool_obj.function_declarations) == 1
    assert tool_obj.function_declarations[0].name == "fake_tool"
    print("\n✅ Verification Passed: Tool object correctly injected into config.")

if __name__ == "__main__":
    asyncio.run(test_client_injects_tools())
