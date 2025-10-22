# filename: app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import os
from datetime import datetime
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# -------------------------
# FastAPI & CORS setup
# -------------------------
app = FastAPI(title="AI Chatbot API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Global Storage & Config
# -------------------------
data_store = {"df": None, "metadata": {}, "data_summary": ""}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyC2aoFncoyYIa56AiH39POKixZn3E_VCus")
if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not configured.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# -------------------------
# Pydantic models
# -------------------------
class ChatMessage(BaseModel):
    message: str
    mode: str = "to-the-point"  # options: to-the-point, detailed, inference


class ChatResponse(BaseModel):
    response: str
    data_summary: Optional[dict] = None
    timestamp: str


# -------------------------
# Data analysis helpers
# -------------------------
def analyze_dataframe(df: pd.DataFrame) -> dict:
    meta = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object']).columns),
    }

    if 'Total Spend' in df.columns:
        meta.update({
            "total_spend": float(df['Total Spend'].sum()),
            "avg_spend": float(df['Total Spend'].mean())
        })

    if 'Quantity Purchased' in df.columns:
        meta["total_quantity"] = float(df['Quantity Purchased'].sum())

    return meta


def create_data_context(df: pd.DataFrame, metadata: dict) -> str:
    return f"""
DATASET OVERVIEW:
Rows: {metadata.get("total_rows")}
Columns: {', '.join(metadata.get("columns", []))}

NUMERIC COLUMNS: {', '.join(metadata.get("numeric_columns", []))}
CATEGORICAL COLUMNS: {', '.join(metadata.get("categorical_columns", []))}
"""


# -------------------------
# Prompt templates
# -------------------------
TO_THE_POINT_TEMPLATE = """You are a professional data analyst assistant. Provide CONCISE, DIRECT answers.
Do NOT output markdown formatting.


{chat_history}

{data_context}

USER QUESTION: {question}

ANSWER:"""

DETAILED_TEMPLATE = """You are an experienced business intelligence analyst. Provide COMPREHENSIVE, DETAILED insights.
Avoid introductory filler phrases; answer directly and professionally.
Avoid discussing data quality issues or dataset errors.
Limit response length to 200-300 words. Provide a structured, detailed explanation (sections, bullets, or short paragraphs).
End with a short "Summary Insight" paragraph highlighting the main takeaway.
Do NOT use markdown formatting.

{chat_history}

{data_context}

USER QUESTION: {question}

DETAILED RESPONSE:"""

INFERENCE_TEMPLATE = """You are a senior data strategist specializing in procurement and supply chain analytics. 
Provide DEEP ANALYTICAL INSIGHTS with calculations. Use a confident, executive tone and conclude with clear action recommendations.
Avoid all introductory or closing phrases. 
Avoid discussing data quality issues or dataset errors.
Keep response concise within 200-300 words.
Do NOT output markdown formatting.

{chat_history}

{data_context}

USER QUESTION: {question}

INFERENCE:"""


# -------------------------
# Generate LLM response
# -------------------------
def get_llm_response(question: str, data_context: str, mode: str) -> str:
    try:
        # Select the right prompt template
        if mode == "detailed":
            template = DETAILED_TEMPLATE
        elif mode == "inference":
            template = INFERENCE_TEMPLATE
        else:
            template = TO_THE_POINT_TEMPLATE

        prompt = PromptTemplate(
            input_variables=["chat_history", "data_context", "question"],
            template=template
        )

        # Get stored memory
        mem_vars = memory.load_memory_variables({})
        chat_history_msgs = mem_vars.get("chat_history", [])

        # Convert memory list into readable text
        chat_history = "\n".join(
            f"{msg.type if hasattr(msg, 'type') else 'Message'}: {msg.content if hasattr(msg, 'content') else str(msg)}"
            for msg in chat_history_msgs
        ) if isinstance(chat_history_msgs, list) else str(chat_history_msgs)

        # Format prompt
        prompt_text = prompt.format(
            chat_history=chat_history,
            data_context=data_context,
            question=question
        )

        # Debug (optional)
        # print("🧩 Prompt Sent to LLM:\n", prompt_text)

        # Invoke LLM directly (no need for RunnableSequence)
        result = llm.invoke(prompt_text)

        # Save chat to memory
        memory.save_context(
            {"question": question},
            {"response": result.content if hasattr(result, "content") else str(result)}
        )

        return result.content if hasattr(result, "content") else str(result)

    except Exception as e:
        return f"Error generating response: {str(e)}"


# -------------------------
# API endpoints
# -------------------------
@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx or .xls)")

    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    data_store["df"] = df
    data_store["metadata"] = analyze_dataframe(df)
    data_store["data_summary"] = create_data_context(df, data_store["metadata"])
    return {"message": "File uploaded and processed", "metadata": data_store["metadata"]}


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="Upload a dataset before chatting")

    response = get_llm_response(
        chat_message.message,
        data_store["data_summary"],
        chat_message.mode
    )

    return ChatResponse(
        response=response,
        data_summary=data_store["metadata"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/reset_memory")
async def reset_memory():
    memory.clear()
    return {"message": "Conversation memory cleared."}


@app.get("/metadata")
async def get_metadata():
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    return data_store["metadata"]


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "data_loaded": data_store["df"] is not None,
        "gemini_ready": bool(GOOGLE_API_KEY)
    }


@app.get("/")
async def root():
    return {
        "message": "Gemini Chatbot with Short-Term Memory",
        "version": "1.1",
        "endpoints": {
            "POST /upload": "Upload Excel file",
            "POST /chat": "Chat with the model",
            "POST /reset_memory": "Reset memory context",
            "GET /metadata": "Get data metadata",
            "GET /health": "Health check"
        }
    }


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI LangChain App with Memory...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
