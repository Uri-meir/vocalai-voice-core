import logging
from src.tools.schemas import (
    ToolContext, 
    ToolResult, 
    GetOpenSlotsArgs, 
    BookAppointmentArgs
)
from datetime import datetime
from src.tools.n8n_client import N8NClient
from src.tools.idempotency import IdempotencyLayer
from src.tools.gates import check_gate
from src.core.events.emitter import get_supabase_vapi_webhook_emitter
import asyncio

logger = logging.getLogger(__name__)

# Shared clients
# In a real app, these might be injected or singletons
n8n_client = N8NClient()
idempotency = IdempotencyLayer()
emitter = get_supabase_vapi_webhook_emitter()

async def get_open_slots_tool(args: GetOpenSlotsArgs, context: ToolContext) -> ToolResult:
    """
    Query n8n for available calendar slots.
    """
    check_gate("getOpenSlots", context)
    
    # Read-only, no idempotency needed theoretically, but caching short term is fine.
    # We pass 'getOpenSlots' as the tool name expected by n8n (from user json)
    return await n8n_client.execute_tool(
        tool_name="getOpenSlots",
        args=args.model_dump(exclude_none=True),
        context_data={
            "call_id": context.call_id,
            "professional_slug": context.professional_slug
        }
    )

async def book_appointment_tool(args: BookAppointmentArgs, context: ToolContext) -> ToolResult:
    """
    Book an appointment via n8n. Idempotent.
    """
    tool_name = "bookAppointment"
    check_gate(tool_name, context)
    
    args_dict = args.model_dump(exclude_none=True)
    
    # 1. Check Idempotency Cache
    cached = idempotency.get_cached_result(context.call_id, tool_name, args_dict)
    if cached:
        return cached

    # 2. Execute
    result = await n8n_client.execute_tool(
        tool_name=tool_name,
        args=args_dict,
        context_data={
            "call_id": context.call_id,
            "professional_slug": context.professional_slug
        }
    )
    
    # 3. Cache Success
    if result.success:
        idempotency.cache_result(context.call_id, tool_name, args_dict, result)
        
        # 4. Emit Meeting Scheduled Event
        if context.assistant_id_webhook:
             # Fire and forget
             asyncio.create_task(
                 emitter.emit_meeting_scheduled(
                     call_id=context.call_id,
                     assistant_id=context.assistant_id_webhook,
                     meeting_details={
                         "tool": tool_name,
                         "args": args_dict,
                         "timestamp": datetime.now().isoformat()
                     }
                 )
             )
    
    return result
