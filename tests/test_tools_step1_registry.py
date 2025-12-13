import pytest
import asyncio
from src.tools.registry import ToolRegistry, ToolContext, ToolResult
from src.tools.schemas import GetOpenSlotsArgs
from pydantic import BaseModel, Field

# Mock Tools
class MockArgs(BaseModel):
    foo: str

async def mock_executor(args: MockArgs, context: ToolContext) -> ToolResult:
    if args.foo == "error":
        raise RuntimeError("Boom")
    return ToolResult.success_result({"result": f"processed {args.foo}"})

async def slow_executor(args: MockArgs, context: ToolContext) -> ToolResult:
    await asyncio.sleep(0.2)
    return ToolResult.success_result({"result": "done"})

@pytest.mark.asyncio
async def test_registry_registration_and_get():
    registry = ToolRegistry()
    
    registry.register(
        name="test_tool",
        description="A test tool",
        args_model=MockArgs
    )(mock_executor)
    
    spec = registry.get_tool("test_tool")
    assert spec is not None
    assert spec.name == "test_tool"
    assert spec.description == "A test tool"
    assert spec.args_model == MockArgs

@pytest.mark.asyncio
async def test_execution_success():
    registry = ToolRegistry()
    registry.register("test_tool", "desc", MockArgs)(mock_executor)
    
    ctx = ToolContext(
        call_id="123", 
        twilio_call_sid="sid", 
        professional_slug="test-slug", 
        state={}
    )
    
    result = await registry.execute("test_tool", {"foo": "bar"}, ctx)
    assert result.success is True
    assert result.data["result"] == "processed bar"

@pytest.mark.asyncio
async def test_execution_validation_error():
    registry = ToolRegistry()
    registry.register("test_tool", "desc", MockArgs)(mock_executor)
    
    ctx = ToolContext(call_id="1", twilio_call_sid="1", professional_slug="1")
    
    # Missing required 'foo'
    result = await registry.execute("test_tool", {}, ctx)
    assert result.success is False
    assert result.error.code == "VALIDATION_ERROR"

@pytest.mark.asyncio
async def test_execution_timeout():
    registry = ToolRegistry()
    registry.register("slow_tool", "desc", MockArgs, timeout=0.1)(slow_executor)
    
    ctx = ToolContext(call_id="1", twilio_call_sid="1", professional_slug="1")
    
    result = await registry.execute("slow_tool", {"foo": "bar"}, ctx)
    assert result.success is False
    assert result.error.code == "TIMEOUT"
    assert result.error.retryable is True

@pytest.mark.asyncio
async def test_gemini_declarations():
    registry = ToolRegistry()
    registry.register("getOpenSlots_uri", "Check slots", GetOpenSlotsArgs)(mock_executor)
    
    decls = registry.get_gemini_declarations()
    assert len(decls) == 1
    tool = decls[0]
    assert tool["name"] == "getOpenSlots_uri"
    assert tool["description"] == "Check slots"
    assert tool["parameters"]["type"] == "OBJECT"
    assert "requestedAppointment" in tool["parameters"]["required"]
