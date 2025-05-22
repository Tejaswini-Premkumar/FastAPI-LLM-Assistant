from pydantic import BaseModel
from typing import Optional

class Ticket(BaseModel):
    subject: str
    description: str
    priority: str
    category: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str

class TicketResponse(BaseModel):
    ticket_id: str
    message: str
    