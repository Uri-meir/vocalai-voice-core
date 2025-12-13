import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.tools.schemas import ToolContext, GetOpenSlotsArgs, BookAppointmentArgs, TransferCallArgs
from src.tools.scheduling import get_open_slots_tool, book_appointment_tool
from src.tools.telephony import transfer_call_tool

@pytest.fixture
def mock_context():
    return ToolContext(
        call_id="test-call-1",
        twilio_call_sid="test-sid-1",
        professional_slug="test-slug",
        state={}
    )

@pytest.mark.asyncio
async def test_get_open_slots_integration(mock_context):
    """Test transparent pass-through to n8n"""
    with patch("src.tools.scheduling.n8n_client.execute_tool", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value.success = True
        mock_exec.return_value.data = {"slots": []}
        
        args = GetOpenSlotsArgs(callerTimezone="UTC", requestedAppointment="2024-01-01")
        result = await get_open_slots_tool(args, mock_context)
        
        assert result.success is True
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        assert call_args.kwargs["tool_name"] == "getOpenSlots_uri"
        assert call_args.kwargs["context_data"]["professional_slug"] == "test-slug"

@pytest.mark.asyncio
async def test_book_appointment_idempotency(mock_context):
    """Test that second call returns cached result without hitting n8n"""
    args = BookAppointmentArgs(name="Uri", callerTimezone="UTC", requestedAppointment="2024-01-01")
    
    with patch("src.tools.scheduling.n8n_client.execute_tool", new_callable=AsyncMock) as mock_exec:
        # First Call
        mock_exec.return_value.success = True
        mock_exec.return_value.data = {"id": "123"}
        
        res1 = await book_appointment_tool(args, mock_context)
        assert res1.success is True
        assert mock_exec.call_count == 1
        
        # Second Call (Same Args)
        res2 = await book_appointment_tool(args, mock_context)
        assert res2.success is True
        assert res2.data["id"] == "123"
        assert mock_exec.call_count == 1 # Should NOT increment

@pytest.mark.asyncio
async def test_transfer_call_gate(mock_context):
    args = TransferCallArgs()
    
    with patch("src.tools.telephony.TwilioClientWrapper") as MockTwilio:
        # 1. Success
        res = await transfer_call_tool(args, mock_context)
        assert res.success is True
        assert mock_context.state["transferred"] is True
        MockTwilio.return_value.transfer_call.assert_called_with("test-sid-1", "+972549182494")
        
        # 2. Gate Block (Already Transferred)
        res = await transfer_call_tool(args, mock_context)
        assert res.success is False
        assert "already been transferred" in res.error.message
