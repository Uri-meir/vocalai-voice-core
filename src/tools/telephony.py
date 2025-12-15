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
        # Use dynamic business phone if available, else fallback provided by caller or config
        target_number = context.business_phone or ESCALATION_NUMBER
        if not target_number:
            raise ValueError("No transfer destination available (business_phone missing)")

        client = TwilioClientWrapper()
        client.transfer_call(context.twilio_call_sid, target_number)
        
        # 3. Update State
        context.state["transferred"] = True
        
        # 4. Emit Tool Call Event (Fire and Forget)
        from src.core.events.emitter import get_supabase_vapi_webhook_emitter
        import asyncio
        
        if context.assistant_id_webhook:
             emitter = get_supabase_vapi_webhook_emitter()
             asyncio.create_task(
                 emitter.emit_tool_call(
                     call_id=context.call_id,
                     assistant_id=context.assistant_id_webhook,
                     customer_number=context.customer_number or "unknown",
                     tool_name="transferCall",
                     tool_args={"destination": target_number},
                     result={"status": "transferred"}
                 )
             )

        return ToolResult.success_result(
            data={"status": "transferred", "destination": target_number},
            meta={"side_effect": True}
        )

    except Exception as e:
        logger.error(f"Transfer failed: {e}")
        return ToolResult.error_result("TRANSFER_FAILED", str(e), retryable=True)
