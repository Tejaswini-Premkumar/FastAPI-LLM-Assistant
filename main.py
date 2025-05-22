# main.py
print("--- main.py: Script started ---") # DEBUG PRINT

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from llm_logic import classify_intent, answer_question, extract_ticket_info
from database import create_db_tables, get_db, Ticket
from sqlalchemy.orm import Session

app = FastAPI()

@app.on_event("startup")
def on_startup():
    print("--- FastAPI Startup Event: Initializing database tables ---") # DEBUG PRINT
    create_db_tables()
    print("--- FastAPI Startup Event: Database initialization complete ---") # DEBUG PRINT

class UserInput(BaseModel):
    query: str
    # Removed session_id: Optional[str] = None
    additional_data: dict = {}

class AnswerResponse(BaseModel):
    query: str
    intent: str
    message: str
    ticket_info: Optional[dict] = None
    answer_info: Optional[dict] = None 


@app.post("/inquiries/", response_model=AnswerResponse)
async def handle_inquiry(user_input: UserInput, db: Session = Depends(get_db)):
    """
    Handles user inquiries, classifying intent, and providing answers or routing to support.
    """
    # Removed print(f"--- handle_inquiry: Request received for session {user_input.session_id} ---")
    print(f"DEBUG: Type of 'db' object received by handle_inquiry: {type(db)}") # DEBUG PRINT

    user_query = user_input.query
    # Removed session_id = user_input.session_id if user_input.session_id else "default_session"

    print(f"Received query: {user_input.query}")
    # Removed print(f"Session ID: {session_id}")

    intent = classify_intent(user_input.query)
    print(f"Intent classified as: '{intent}'")
    
    response_message = ""
    ticket_info = None
    answer_info = None

    if intent == "product_question":
        print(f"Intent classified as 'product_question'. Answering: {user_input.query}")
        answer_info = answer_question(user_input.query) 
        response_message = answer_info["answer"]
        
    elif intent == "technical_issue" or intent == "support_ticket_request":
        print(f"Intent classified as '{intent}'. Creating ticket for: {user_input.query}")
        ticket_info_extracted = extract_ticket_info(user_input.model_dump())
        
        new_ticket = Ticket(
            subject=ticket_info_extracted.get("subject", "N/A"),
            description=ticket_info_extracted.get("description", user_query),
            product_name=ticket_info_extracted.get("product_name", "N/A"),
            user_id=ticket_info_extracted.get("user_id", "N/A"),
            priority=ticket_info_extracted.get("priority", "Medium"),
            extracted_status=ticket_info_extracted.get("extracted_status", "success")
        )
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)

        ticket_info = ticket_info_extracted
        response_message = f"Ticket #{new_ticket.id} created with subject: {new_ticket.subject}. We will get back to you shortly."
        print(f"Ticket saved to database: {new_ticket}")

    elif intent == "general_greeting":
        print(f"Intent classified as 'general_greeting'. Responding with greeting.")
        response_message = "Hello! How can I assist you today regarding our products?"

    elif intent == "order_status_inquiry":
        print(f"Intent classified as 'order_status_inquiry'. Prompting for order number.")
        response_message = "I can help with order status. Please provide your order number."

    elif intent == "return_refund_query":
        print(f"Intent classified as 'return_refund_query'. Guiding to returns process.")
        response_message = "For returns or refunds, please visit our returns policy page or provide more details like your order ID."

    else:
        print(f"Warning: Unhandled intent '{intent}'. Returning generic response.")
        response_message = "I'm not sure how to handle that request. Could you please rephrase or ask about Intoleads products or services?"

    return AnswerResponse(
        query=user_input.query,
        intent=intent,
        message=response_message,
        ticket_info=ticket_info,
        answer_info=answer_info
    )

@app.get("/tickets/", response_model=List[Dict])
async def get_all_tickets(db: Session = Depends(get_db)):
    print(f"--- get_all_tickets: Request received ---") # DEBUG PRINT
    print(f"DEBUG: Type of 'db' object received by get_all_tickets: {type(db)}") # DEBUG PRINT
    tickets = db.query(Ticket).all()
    return [
        {
            "id": t.id,
            "subject": t.subject,
            "description": t.description,
            "product_name": t.product_name,
            "user_id": t.user_id,
            "priority": t.priority,
            "extracted_status": t.extracted_status,
            "created_at": t.created_at.isoformat()
        } for t in tickets
    ]


if __name__ == "__main__":
    print("--- main.py: Running Uvicorn ---") # DEBUG PRINT
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)