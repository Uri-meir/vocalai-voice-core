import httpx
import logging
from src.tools.schemas import ToolContext, ToolResult, SendWhatsAppArgs
from src.config.environment import config

logger = logging.getLogger(__name__)

async def send_whatsapp_tool(args: SendWhatsAppArgs, context: ToolContext) -> ToolResult:
    """
    Sends customer data to n8n webhook for WhatsApp messaging.
    
    Args:
        args.name: Customer name (from agent conversation)
        context.customer_number: Caller phone number (from Twilio)
    """
    # Get webhook URL from environment via ConfigManager property
    webhook_url = config.WHATSAPP_WEBHOOK_URL
    
    if not webhook_url:
        logger.error("❌ WHATSAPP_WEBHOOK_URL not set in environment or config")
        return ToolResult.error_result("CONFIG_ERROR", "WhatsApp webhook URL not configured", retryable=False)
    
    payload = {
        "name": args.name,
        "customer_number": context.customer_number,
        "business_phone": context.business_phone,
        "business_owner_name": context.business_owner_name,
        "professional_slug": context.professional_slug,
        "assistant_id": context.assistant_id_webhook,
        "call_sid": context.twilio_call_sid,
        "call_id": context.call_id
    }
    
    logger.info(f"📱 Sending WhatsApp data to n8n ({webhook_url}): {payload}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"✅ WhatsApp webhook succeeded: {response.text}")
                return ToolResult.success_result(
                    data={"status": "sent", "message": "WhatsApp message queued successfully"}
                )
            else:
                logger.error(f"❌ WhatsApp webhook failed: {response.status_code} - {response.text}")
                return ToolResult.error_result(
                    "WEBHOOK_ERROR",
                    f"n8n returned {response.status_code}",
                    retryable=response.status_code >= 500
                )
                
    except httpx.TimeoutException:
        logger.error("⏱️ WhatsApp webhook timeout")
        return ToolResult.error_result("TIMEOUT", "Webhook request timed out", retryable=True)
        
    except Exception as e:
        logger.exception("💥 WhatsApp webhook error")
        return ToolResult.error_result("EXECUTION_ERROR", str(e), retryable=False)
