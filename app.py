# filename: app.py
import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import os
from datetime import datetime
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain



# -------------------------
# FastAPI App & CORS Setup
# -------------------------
app = FastAPI(title="AI Chatbot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Global Storage
# -------------------------
data_store = {"df": None, "metadata": {}, "data_summary": ""}

# -------------------------
# Initialize Gemini LLM
# -------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAi_CD0PcVCum24YRC1UnRm0_QLTQeS4m0")
if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not set. Please set it as an environment variable.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# -------------------------
# Pydantic Models
# -------------------------
class ChatMessage(BaseModel):
    message: str
    mode: str = "to-the-point"  # options: to-the-point, detailed, inference

class ChatResponse(BaseModel):
    response: str
    data_summary: Optional[dict] = None
    timestamp: str

# -------------------------
# Data Analysis Functions
# -------------------------
def analyze_dataframe(df: pd.DataFrame) -> dict:
    metadata = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object']).columns),
    }
    
    if 'Total Spend' in df.columns:
        metadata.update({
            "total_spend": float(df['Total Spend'].sum()),
            "avg_spend": float(df['Total Spend'].mean()),
            "max_spend": float(df['Total Spend'].max()),
            "min_spend": float(df['Total Spend'].min())
        })
    
    if 'Quantity Purchased' in df.columns:
        metadata.update({
            "total_quantity": float(df['Quantity Purchased'].sum()),
            "avg_quantity": float(df['Quantity Purchased'].mean())
        })
    
    if 'Commodity' in df.columns:
        metadata.update({
            "unique_commodities": df['Commodity'].nunique(),
            "commodities_list": df['Commodity'].unique().tolist()
        })
    
    if 'Top Supplier' in df.columns:
        metadata.update({
            "unique_suppliers": df['Top Supplier'].nunique(),
            "suppliers_list": df['Top Supplier'].unique().tolist()
        })
    
    if 'Total Spend' in df.columns and 'Quantity Purchased' in df.columns:
        df_temp = df.copy()
        df_temp['cost_per_unit'] = df_temp['Total Spend'] / df_temp['Quantity Purchased']
        metadata.update({
            "avg_cost_per_unit": float(df_temp['cost_per_unit'].mean()),
            "max_cost_per_unit": float(df_temp['cost_per_unit'].max()),
            "min_cost_per_unit": float(df_temp['cost_per_unit'].min())
        })
    
    return metadata


def create_data_context(df: pd.DataFrame, metadata: dict) -> str:
    context = f"""
DATA CONTEXT:
- Total Records: {metadata.get('total_rows', 0)}
- Columns: {', '.join(metadata.get('columns', []))}

KEY METRICS:
- Total Spend: ${metadata.get('total_spend', 0):,.2f}
- Total Quantity: {metadata.get('total_quantity', 0):,.2f} units
- Number of Commodities: {metadata.get('unique_commodities', 0)}
- Number of Suppliers: {metadata.get('unique_suppliers', 0)}
- Average Cost per Unit: ${metadata.get('avg_cost_per_unit', 0):.2f}

AVAILABLE DATA SAMPLE (First 5 rows):
{df.head().to_string()}

SUMMARY STATISTICS:
{df.describe().to_string()}
"""
    
    # Commodity breakdown
    if 'Commodity' in df.columns and 'Total Spend' in df.columns:
        commodity_spend = df.groupby('Commodity')['Total Spend'].sum().sort_values(ascending=False)
        context += "\n\nCOMMODITY SPEND BREAKDOWN:\n"
        for comm, spend in commodity_spend.items():
            context += f"- {comm}: ${spend:,.2f}\n"
    
    # Supplier breakdown
    if 'Top Supplier' in df.columns and 'Total Spend' in df.columns:
        supplier_spend = df.groupby('Top Supplier')['Total Spend'].sum().sort_values(ascending=False).head(5)
        context += "\n\nTOP 5 SUPPLIERS BY SPEND:\n"
        for supplier, spend in supplier_spend.items():
            context += f"- {supplier}: ${spend:,.2f}\n"
    
    return context

# -------------------------
# Prompt Templates
# -------------------------
TO_THE_POINT_TEMPLATE = """You are a professional data analyst assistant. Provide CONCISE, DIRECT answers.

{data_context}

USER QUESTION: {question}

CONCISE ANSWER:"""

DETAILED_TEMPLATE = """You are an experienced business intelligence analyst. Provide COMPREHENSIVE, DETAILED insights.
Avoid introductory filler phrases; answer directly and professionally.
Avoid discussing data quality issues or dataset errors.
Limit response length to 200-300 words.
Do NOT use markdown formatting.

{data_context}

USER QUESTION: {question}

DETAILED ANALYSIS:"""


INFERENCE_TEMPLATE = """You are a senior data strategist specializing in procurement and supply chain analytics. Provide DEEP ANALYTICAL INSIGHTS with calculations.
Avoid all introductory or closing phrases.
Avoid discussing data quality issues or dataset errors.
Keep response concise within 200-300 words.
Do NOT output markdown formatting.

{data_context}

USER QUESTION: {question}

ANALYTICAL INSIGHTS:"""


def get_llm_response(question: str, data_context: str, mode: str) -> str:
    if mode == "detailed":
        template = DETAILED_TEMPLATE
    elif mode == "inference":
        template = INFERENCE_TEMPLATE
    else:
        template = TO_THE_POINT_TEMPLATE
    
    prompt = PromptTemplate(input_variables=["data_context", "question"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        return chain.run({"data_context": data_context, "question": question}).strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# -------------------------
# FastAPI Endpoints
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
    except Exception:
        df = pd.read_excel(io.BytesIO(contents))
    
    data_store["df"] = df
    data_store["metadata"] = analyze_dataframe(df)
    data_store["data_summary"] = create_data_context(df, data_store["metadata"])
    
    return {
        "message": "File uploaded and analyzed successfully",
        "metadata": data_store["metadata"],
        "row_count": len(df),
        "columns": df.columns.tolist()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No data available. Upload Excel file first.")
    
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured.")
    
    response_text = get_llm_response(
        chat_message.message,
        data_store["data_summary"],
        chat_message.mode
    )
    
    return ChatResponse(
        response=response_text,
        data_summary=data_store["metadata"],
        timestamp=datetime.now().isoformat()
    )

@app.get("/metadata")
async def get_metadata():
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No data available")
    return data_store["metadata"]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": data_store["df"] is not None,
        "gemini_configured": bool(GOOGLE_API_KEY)
    }

@app.get("/")
async def root():
    return {
        "message": "AI Chatbot API with Gemini",
        "version": "1.0",
        "endpoints": {
            "POST /upload": "Upload Excel file",
            "POST /chat": "Send chat message with mode toggle",
            "GET /metadata": "Get uploaded data metadata",
            "GET /health": "Health check"
        }
    }

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Chatbot Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
