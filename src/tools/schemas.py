from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

class ToolContext(BaseModel):
    """
    Context passed to every tool execution.
    Contains per-call state and clients.
    """
    call_id: str
    twilio_call_sid: str
    professional_slug: str
    assistant_id_webhook: Optional[str] = None
    caller_timezone: str = "Asia/Jerusalem"
    # State is a mutable dictionary for lifecycle gates (e.g. transfer_initiated)
    # Using Any to avoid circular imports if we pass complex objects, but ideally dict.
    state: Dict[str, Any] = Field(default_factory=dict) 
    
    class Config:
        arbitrary_types_allowed = True

class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False

class ToolResult(BaseModel):
    """
    Standard output envelope for all tools.
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[ToolError] = None
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def success_result(cls, data: Dict[str, Any], meta: Dict[str, Any] = None):
        return cls(success=True, data=data, meta=meta)

    @classmethod
    def error_result(cls, code: str, message: str, retryable: bool = False):
        return cls(
            success=False, 
            error=ToolError(code=code, message=message, retryable=retryable)
        )

# --- Tool Input Models ---

class GetOpenSlotsArgs(BaseModel):
    callerTimezone: str = Field(..., description="IANA timezone such as 'Asia/Jerusalem'.")
    requestedAppointment: str = Field(..., description="Appointment time in ISO 8601 with offset, e.g. '2025-12-08T14:00:00+02:00'.")

class BookAppointmentArgs(BaseModel):
    name: str = Field(..., description="Caller full name.")
    callerTimezone: str = Field(..., description="IANA timezone such as 'Asia/Jerusalem'.")
    requestedAppointment: str = Field(..., description="Appointment time in ISO 8601 with offset, e.g. '2025-12-08T11:30:00+02:00'.")

class TransferCallArgs(BaseModel):
    pass
