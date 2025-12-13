from pydantic import BaseModel, Field, AliasChoices, UUID4
from typing import Optional, Dict

class CustomerData(BaseModel):
    number: str

class StartCallRequest(BaseModel):
    customer: CustomerData
    assistantId: str = Field(..., alias="assistantId") 
    phoneNumberId: Optional[str] = None

    @property
    def to(self) -> str:
        return self.customer.number

    @property
    def assistant_id(self) -> str:
        return self.assistantId

class RegisterPhoneNumberRequest(BaseModel):
    assistantId: UUID4
    professionalSlug: str
    phoneNumber: str
    twilioAccountSid: Optional[str] = None
