import logging
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone
from src.api.models import RegisterPhoneNumberRequest
from src.config.supabase_client import get_supabase_client

router = APIRouter(tags=["phone_numbers"])
logger = logging.getLogger(__name__)

@router.post("/phone-number/register", status_code=status.HTTP_200_OK)
async def register_phone_number(payload: RegisterPhoneNumberRequest):
    logger.info(f"📝 Registering phone number: {payload.phoneNumber} for slug: {payload.professionalSlug}")

    try:
        supabase = get_supabase_client()
        
        # 1. Validation Logic
        assistant_id_str = str(payload.assistantId)
        
        # A. Check voice_assistant_configs (Primary Definition)
        config_resp = supabase.table("voice_assistant_configs")\
            .select("professional_slug")\
            .eq("id", assistant_id_str)\
            .execute()
            
        validated = False
        
        if config_resp.data:
            # Found in configs, check ownership
            config_slug = config_resp.data[0].get("professional_slug")
            if config_slug == payload.professionalSlug:
                validated = True
            else:
                logger.warning(f"❌ Slug mismatch in config: {config_slug} != {payload.professionalSlug}")
                # We fail hard here if found but wrong slug
                raise HTTPException(
                     status_code=status.HTTP_400_BAD_REQUEST,
                     detail="assistantId does not belong to the provided professionalSlug"
                 )
        
        # B. Fallback: Check assistants table (Existing Assignment)
        # If not found in configs (or maybe legacy ID), check if it's already assigned in assistants
        if not validated:
            assistants_check = supabase.table("assistants")\
                .select("*")\
                .eq("professional_slug", payload.professionalSlug)\
                .execute()
            
            if assistants_check.data:
                row = assistants_check.data[0]
                # Check internal_assistant_id
                if row.get("internal_assistant_id") == assistant_id_str:
                    logger.info(f"✅ Assistant ID {assistant_id_str} validated against existing internal_assistant_id")
                    validated = True
                # Check vapi_assistant_id (legacy column)
                elif row.get("vapi_assistant_id") == assistant_id_str:
                    logger.info(f"✅ Assistant ID {assistant_id_str} validated against existing vapi_assistant_id")
                    validated = True
                    
        if not validated:
             logger.warning(f"❌ Assistant ID not found in configs or current assignment: {payload.assistantId}")
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail="assistantId not found or not valid for this professional"
             )

        # 2. Check current state of assistants table for updating
        # We might have already fetched it in Step 1B, but to keep logic clean and robust:
        assistants_resp = supabase.table("assistants")\
            .select("phone_number_created_at")\
            .eq("professional_slug", payload.professionalSlug)\
            .execute()
            
        if not assistants_resp.data:
             logger.warning(f"❌ Assistant row not found for slug: {payload.professionalSlug}")
             raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail="Assistant row not found for professionalSlug"
             )
        
        current_created_at = assistants_resp.data[0].get("phone_number_created_at")
        
        # 3. Prepare Update Payload
        now = datetime.now(timezone.utc).isoformat()
        
        update_data = {
            "phone_number_e164": payload.phoneNumber,
            "twilio_did_e164": payload.phoneNumber,
            "internal_assistant_id": str(payload.assistantId),
            "twilio_account_sid": payload.twilioAccountSid,
            "updated_at": now
        }
        
        # Update phone_number_created_at ONLY if it is currently None/Null
        if not current_created_at:
            update_data["phone_number_created_at"] = now
        
        logger.debug(f"Update payload: {update_data}")

        # 4. Execute Update
        update_resp = supabase.table("assistants")\
            .update(update_data)\
            .eq("professional_slug", payload.professionalSlug)\
            .execute()
            
        if not update_resp.data:
            # Should not happen given step 2 passed, but safe guard
            raise HTTPException(status_code=404, detail="Assistant row not found during update")

        logger.info(f"✅ Successfully updated assistant: {payload.assistantId}")
        
        return {
            "status": "success",
            "professionalSlug": payload.professionalSlug,
            "assistantId": payload.assistantId,
            "phoneNumber": payload.phoneNumber
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Supabase error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Database Error"
        )
