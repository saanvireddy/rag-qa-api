from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.db import init_db
from app.rag import RAGPipeline
import os
import time

app = FastAPI(
    title="RAG Q&A API",
    description="Production-ready RAG service"
)

rag = RAGPipeline()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    latency_ms: float

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print("✅ API Started - RAG Pipeline Ready")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Q&A API"
    }

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    try:
        # Save temporary file
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Check if PDF
        if not file.filename.endswith(".pdf"):
            os.remove(temp_path)
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Load and process documents
        documents = rag.load_documents(temp_path)
        rag.store_documents(documents)
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_processed": len(documents)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Query documents and get RAG-generated answer"""
    try:
        # Validate input
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Execute RAG query
        answer, sources, latency = rag.query(request.question)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency_ms=round(latency, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))