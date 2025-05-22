from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional,List, Dict
from llm_logic import classify_intent, answer_question, extract_ticket_info
app = FastAPI()
class UserInput(BaseModel):
    query: str
    #session_id: Optional[str] = None
    additional_data: dict = {}

class AnswerResponse(BaseModel):
    query: str
    intent: str
    #answer: str
    #sources: List[str] = [] 
    message: str
    ticket_info: Optional[dict] = None
    answer_info: Optional[dict] = None 


@app.post("/inquiries/", response_model=AnswerResponse)
async def handle_inquiry(user_input: UserInput):
    """
    Handles user inquiries, either answering questions or creating support tickets.
    """
    print(f"Received query: {user_input.query}") # For debugging
    intent = classify_intent(user_input.query)
    print(f"Intent classified as: '{intent}'")
    response_data = {"query": user_input.query, "intent": intent}
    #"""
    #Handles user inquiries, managing conversation history, classifying intent,
    #and providing answers or routing to support.
    #"""
    user_query = user_input.query
    #session_id = user_input.session_id if user_input.session_id else "default_session" # Use a default if not provided

    print(f"Received query: {user_input.query}")
    #print(f"Session ID: {session_id}")

    # Get or create conversation memory for the session
    #if session_id not in conversation_memories:
    #    print(f"Creating new session memory for ID: {session_id}")
        # memory_key is the key by which the history will be exposed in the prompt
    #    conversation_memories[session_id] = ConversationBufferMemory(memory_key="chat_history")
    
    #memory = conversation_memories[session_id]
    
    # Load current chat history string from memory
    # memory.load_memory_variables({}) returns {'chat_history': 'Human: ...\nAI: ...'}
    #chat_history_str = memory.load_memory_variables({})["chat_history"] 
    #print(f"Current Chat History for {session_id}:\n{chat_history_str if chat_history_str else ' (empty)'}")

    intent = classify_intent(user_input.query)
    print(f"Intent classified as: '{intent}'")
    
    response_message = ""
    ticket_info = None
    answer_info = None
    if intent == "product_question":
        print(f"Intent classified as 'product_question'. Answering: {user_input.query}")
        answer_info = answer_question(user_input.query)
        response_message = answer_info["answer"]
        
    elif intent == "technical_issue":
        print(f"Intent classified as 'technical_issue'. Suggesting support.")
        ticket_info = extract_ticket_info(user_input.model_dump())
        response_message = "It sounds like a technical issue. I'm routing this to support. Could you provide more details?"

    elif intent == "support_ticket_request":
        print(f"Intent classified as 'support_ticket_request'. Creating ticket for: {user_input.query}")
        ticket_info = extract_ticket_info(user_input.model_dump())
        response_message = f"Ticket created with subject: {ticket_info.get('subject', 'Placeholder Subject')}"

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

    # Save current interaction to memory
    #memory.save_context({"input": user_query}, {"output": response_message})
    
    return AnswerResponse(
        query=user_input.query,
        intent=intent,
        message=response_message,
        ticket_info=ticket_info,
        answer_info=answer_info
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    