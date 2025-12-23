import logging
from src.tools.schemas import (
    ToolContext, 
    ToolResult, 
    GetOpenSlotsArgs, 
    BookAppointmentArgs
)
from datetime import datetime
from typing import Any, Tuple, Optional
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

def resolve_service(args: Any, context: ToolContext) -> Tuple[str, Optional[int]]:
    """
    Resolves the correct event_type_slug and duration based on duration mapping.
    Resolution Order:
    1. Resolve Duration (args -> service lookup -> legacy fallback)
    2. Resolve Slug (duration map -> legacy fallback)
    """
    duration = None
    
    # 1. Resolve Duration
    if args.duration_minutes:
        duration = args.duration_minutes
    elif args.service_name and context.services:
        # Lookup in services by name
        for s in context.services:
            # ServiceConfig is dict here
            if args.service_name.lower() in s.get("name", "").lower():
                duration = s.get("duration")
                logger.info(f"resolve_service: Resolved name '{args.service_name}' to duration {duration}")
                break
    
    # 2. Resolve Cal.com Event Type Slug
    final_slug = None
    
    if duration:
        # Try to map duration to slug
        # event_types_by_duration is Dict[str, Dict[str, str]]
        # e.g. "30": {"type_slug": "..."}
        mapping = context.event_types_by_duration.get(str(duration))
        if mapping:
            final_slug = mapping.get("type_slug")
            logger.info(f"resolve_service: Mapped duration {duration} to slug '{final_slug}'")
    
    # Fallback to legacy behavior if no slug resolved via map
    if not final_slug:
        if args.event_type_slug:
            final_slug = args.event_type_slug
            logger.info(f"resolve_service: Using explicit legacy arg slug '{final_slug}'")
        else:
            final_slug = context.event_type_slug
            logger.info(f"resolve_service: Using context default slug '{final_slug}'")
            
    return final_slug, duration

async def get_open_slots_tool(args: GetOpenSlotsArgs, context: ToolContext) -> ToolResult:
    """
    Query n8n for available calendar slots.
    """
    check_gate("getOpenSlots", context)
    
    resolved_slug, resolved_duration = resolve_service(args, context)
    if not resolved_slug:
        return ToolResult.error_result("MISSING_SERVICE_TYPE", "Could not resolve a valid service type (slug). Please specify duration or service name.")

    # Read-only, no idempotency needed theoretically, but caching short term is fine.
    # We pass 'getOpenSlots' as the tool name expected by n8n (from user json)
    return await n8n_client.execute_tool(
        tool_name="getOpenSlots",
        args=args.model_dump(exclude_none=True),
        context_data={
            "call_id": context.call_id,
            "professional_slug": context.professional_slug,
            "cal_username": context.cal_username,
            "event_type_slug": resolved_slug,
            "duration_minutes": resolved_duration, # Explicitly pass resolved duration
            "cal_api_key": context.cal_api_key,
            "customer_number": context.customer_number
        }
    )

async def book_appointment_tool(args: BookAppointmentArgs, context: ToolContext) -> ToolResult:
    """
    Book an appointment via n8n. Idempotent.
    """
    tool_name = "bookAppointment"
    check_gate(tool_name, context)
    
    resolved_slug, resolved_duration = resolve_service(args, context)
    if not resolved_slug:
         return ToolResult.error_result("MISSING_SERVICE_TYPE", "Could not resolve a valid service type (slug). Please specify duration or service name.")

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
            "professional_slug": context.professional_slug,
            "cal_username": context.cal_username,
            "event_type_slug": resolved_slug,
            "duration_minutes": resolved_duration, # Explicitly pass resolved duration
            "cal_api_key": context.cal_api_key,
            "customer_number": context.customer_number,
            "business_phone": context.business_phone,
            "buissnes_owner": context.business_owner_name
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
                         "slug": resolved_slug,
                         "duration": resolved_duration,
                         "timestamp": datetime.now().isoformat()
                     }
                 )
             )
    
    return result
