import logging
from src.tools.schemas import ToolContext, ToolResult, TransferCallArgs
from src.telephony.twilio_client import TwilioClientWrapper
from src.tools.gates import check_gate

logger = logging.getLogger(__name__)

# Hardcoded destination for now per requirements
ESCALATION_NUMBER = "+972549182494"

async def transfer_call_tool(args: TransferCallArgs, context: ToolContext) -> ToolResult:
    """
    Transfers the current call to a human agent.
    """
    try:
        # 1. Gate Check
        # Ensure we haven't already transferred
        check_gate("transfer_call_tool", context)
        
        logger.info(f"📞 Initiating transfer for call {context.twilio_call_sid}")
        
        # 2. Execute Transfer (Side Effect)
        client = TwilioClientWrapper()
        client.transfer_call(context.twilio_call_sid, ESCALATION_NUMBER)
        
        # 3. Update State
        context.state["transferred"] = True
        
        return ToolResult.success_result(
            data={"status": "transferred", "destination": ESCALATION_NUMBER},
            meta={"side_effect": True}
        )

    except Exception as e:
        logger.error(f"Transfer failed: {e}")
        return ToolResult.error_result("TRANSFER_FAILED", str(e), retryable=True)
