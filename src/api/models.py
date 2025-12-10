from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Customer(BaseModel):
    number: str = Field(..., description="Customer phone number (E.164)")

class StartCallRequest(BaseModel):
    assistantId: str = Field(..., description="Logical assistant identifier")
    phoneNumberId: str = Field(..., description="Phone Number ID (UUID)")
    customer: Customer = Field(..., description="Customer details")
    
    # Optional metadata not in snippet but good to keep
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata")

class StartCallResponse(BaseModel):
    success: bool
    callId: str = Field(..., description="Twilio CallSid")
    status: str = Field(..., description="Call status, e.g. queued, ringing, in-progress, error")
    message: str
