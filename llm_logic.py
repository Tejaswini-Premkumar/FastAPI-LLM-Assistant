from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from document_processor import VECTOR_STORE
import json
import re 

# Initialize the Hugging Face Zero-Shot Classification pipeline
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device='cpu')
except Exception as e:
    print(f"Error loading Hugging Face classifier model: {e}")
    print("Please ensure you have an active internet connection or the model is cached.")
    classifier = None

CANDIDATE_LABELS = [
    "product question",
    "support ticket request",
    "general greeting/small talk",
    "order status inquiry",
    "return/refund query",
    "technical issue"
]

def classify_intent(user_input: str):
    if classifier is None:
        print("Hugging Face classifier not loaded, falling back to dummy intent classification.")
        user_input_lower = user_input.lower()
        if "ticket" in user_input_lower or "support" in user_input_lower or "issue" in user_input_lower:
            return "support_ticket_request"
        elif "order" in user_input_lower or "status" in user_input_lower:
            return "order_status_inquiry"
        elif "return" in user_input_lower or "refund" in user_input_lower:
            return "return_refund_query"
        elif any(greeting in user_input_lower for greeting in ["hello", "hi", "hey"]):
            return "general_greeting"
        elif "tech" in user_input_lower or "problem" in user_input_lower:
            return "technical_issue"
        return "product_question"
    try:
        prediction = classifier(user_input, CANDIDATE_LABELS)
        predicted_label = prediction['labels'][0]
        if "product question" in predicted_label:
            return "product_question"
        elif "support ticket request" in predicted_label:
            return "support_ticket_request"
        elif "general greeting/small talk" in predicted_label:
            return "general_greeting"
        elif "order status inquiry" in predicted_label:
            return "order_status_inquiry"
        elif "return/refund query" in predicted_label:
            return "return_refund_query"
        elif "technical issue" in predicted_label:
            return "technical_issue"
        else:
            print(f"Warning: Model predicted '{predicted_label}', defaulting to 'product_question'.")
            return "product_question"
    except Exception as e:
        print(f"Error during intent classification: {e}")
        user_input_lower = user_input.lower()
        if "ticket" in user_input_lower or "support" in user_input_lower or "issue" in user_input_lower:
            return "support_ticket_request"
        elif "order" in user_input_lower or "status" in user_input_lower:
            return "order_status_inquiry"
        elif "return" in user_input_lower or "refund" in user_input_lower:
            return "return_refund_query"
        elif any(greeting in user_input_lower for greeting in ["hello", "hi", "hey"]):
            return "general_greeting"
        elif "tech" in user_input_lower or "problem" in user_input_lower:
            return "technical_issue"
        return "product_question"


def get_llm_model():
    """
    Initializes and returns a Hugging Face generative LLM for Q&A and extraction.
    """
    try:
        model_id = "google/flan-t5-base" # Using base model instead of small for better performance
        # model_id = "google/flan-t5-small" # Using small model for faster inference
        qa_tokenizer = AutoTokenizer.from_pretrained(model_id)
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        qa_pipeline = pipeline(
            "text2text-generation",
            model=qa_model,
            tokenizer=qa_tokenizer,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            device='cpu'
        )
        llm_instance = HuggingFacePipeline(pipeline=qa_pipeline)
        return llm_instance
    except Exception as e:
        print(f"Error loading Hugging Face QA model: {e}")
        return None

llm = get_llm_model()

QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"])


def answer_question(question: str) -> dict:
    """
    Answers product questions using Retrieval Augmented Generation (RAG)
    from the loaded document and includes source information.
    """
    if VECTOR_STORE is None:
        print("Vector store not initialized. Cannot answer questions.")
        return {"answer": "Apologies, I cannot retrieve product information at this moment.", "sources": []}

    if llm is None:
        print("Hugging Face QA model not loaded. Cannot answer questions.")
        return {"answer": "Apologies, I cannot answer questions at this moment due to an LLM issue.", "sources": []}

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=VECTOR_STORE.as_retriever(),
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )

        response_dict = qa_chain.invoke({"query": question})

        response_text = response_dict.get("result", "")
        source_docs = response_dict.get("source_documents", [])

        sources_content = [doc.page_content for doc in source_docs]

        if "i don't know" in response_text.lower() or "not found" in response_text.lower() or "cannot answer based on the provided context" in response_text.lower():
            final_answer = "I'm sorry, I couldn't find information about that in the provided product details. Could you please rephrase your question or ask about a different feature?"
            return {"answer": final_answer, "sources": []}

        else:
            return {"answer": response_text, "sources": sources_content}

    except Exception as e:
        print(f"Error during question answering: {e}")
        return {"answer": "Apologies, an error occurred while trying to answer your question.", "sources": []}


def extract_ticket_info(user_input_dict: dict) -> dict:
    """
    Extracts structured ticket information from the user's input using the LLM.
    Expected to receive the full user_input_dict which includes 'query'.
    """
    if llm is None:
        print("Hugging Face QA model (LLM) not loaded. Cannot extract ticket info.")
        return {"subject": "LLM Not Available", "description": user_input_dict.get("query", ""), "priority": "Medium", "extracted_status": "failed_llm_load"}

    user_query = user_input_dict.get("query", "")
    if not user_query:
        return {"subject": "No Query Provided", "description": "", "priority": "Medium", "extracted_status": "no_query"}

    extraction_prompt_template = """Extract the following information from the user's request.
Provide the information in a simple key-value pair format, one per line.
If a piece of information is not found, state "N/A" for its value.


Example 1:
User Request: My laptop's webcam is not working after the last update. I have an Intoleads ProBook. My user ID is support_user_001. This is high urgency.
Your Output:
Subject: Webcam not working after update
Description: Laptop webcam stopped functioning after the recent system update.
Product Name: Intoleads ProBook
User ID: support_user_001
Urgency/Priority: High

Example 2:
User Request: The new software isn't installing on my PC.
Your Output:
Subject: Software installation issue
Description: The new software is failing to install on my personal computer.
Product Name: N/A
User ID: N/A
Urgency/Priority: Medium

Format:
Subject: [summary of the issue]
Description: [detailed explanation]
Product Name: [name of product]
User ID: [user ID]
Urgency/Priority: [High, Medium, or Low]

User Request: {request}

Your Output:
"""

    extraction_prompt = PromptTemplate(
        template=extraction_prompt_template,
        input_variables=["request"]
    )

    extracted_data = {
        "subject": "N/A",
        "description": user_query, # Default to full query if extraction fails
        "product_name": "N/A",
        "user_id": "N/A",
        "priority": "Medium"
    }

    try:
        from langchain.chains import LLMChain
        extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)

        raw_llm_output = extraction_chain.run(user_query)
        print(f"Raw LLM output for ticket extraction: {raw_llm_output}")

        # Process the raw LLM output to extract structured information
        # Split the output into lines and parse each line
        lines = raw_llm_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Subject:"):
                extracted_data["subject"] = line.replace("Subject:", "").strip()
            elif line.startswith("Description:"):
                extracted_data["description"] = line.replace("Description:", "").strip()
            elif line.startswith("Product Name:"):
                extracted_data["product_name"] = line.replace("Product Name:", "").strip()
            elif line.startswith("User ID:"):
                extracted_data["user_id"] = line.replace("User ID:", "").strip()
            elif line.startswith("Urgency/Priority:"):
                priority_val= line.replace("Urgency/Priority:", "").strip().lower()
                if "high" in priority_val:
                    extracted_data["priority"] = "High"
                elif "low" in priority_val:
                    extracted_data["priority"] = "Low"
                else:
                    extracted_data["priority"] = "Medium"
            if extracted_data["description"] == "N/A" and user_query:
                extracted_data["description"] = user_query

        if extracted_data["description"] == "N/A" or not extracted_data["description"]:
            extracted_data["description"] = user_query

        if extracted_data["subject"] == "N/A" or "[name of product]" in extracted_data["subject"].lower():
            extracted_data["subject"] = "Ticket: " + user_query[:50] + ("..." if len(user_query) > 50 else "")
        
        if extracted_data["product_name"] == "N/A":
            query_lower = user_query.lower()
            if "intoleads probook" in query_lower:
                extracted_data["product_name"] = "Intoleads ProBook"
            elif "intoleads x1 carbon" in query_lower:
                extracted_data["product_name"] = "Intoleads X1 Carbon"
            #more specific product name checks here if needed

        if extracted_data["user_id"] == "N/A":
            user_id_match = re.search(r'(user id|userid|my id is)\s*(\w+)', user_query, re.IGNORECASE)
            if user_id_match:
                extracted_data["user_id"] = user_id_match.group(2)
        
        if extracted_data["priority"] == "Medium": # Only if default Medium from LLM
            query_lower = user_query.lower()
            if "urgent" in query_lower or "critical" in query_lower or "high priority" in query_lower:
                extracted_data["priority"] = "High"
            elif "low priority" in query_lower:
                extracted_data["priority"] = "Low"
        
        

        final_ticket_info = {
            "subject": extracted_data.get("subject", "N/A"),
            "description": extracted_data.get("description", user_query),
            "product_name": extracted_data.get("product_name", "N/A"),
            "user_id": extracted_data.get("user_id", "N/A"),
            "priority": extracted_data.get("priority", "Medium")
        }
        final_ticket_info["extracted_status"] = "success"
        return final_ticket_info

    except Exception as e:
        print(f"Error during ticket information extraction: {e}")
        return {
            "subject": "Extraction Error - " + user_query[:50] + ("..." if len(user_query) > 50 else ""),
            "description": user_query,
            "product_name": "N/A",
            "user_id": "N/A",
            "priority": "Medium",
            "extracted_status": "general_error"
        }