# main.py - FastAPI application for JengaVest AI backend

import os
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import get_agent
from rag import initialize_rag_pipeline
from tools import TOOLS

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────────────────
# REQUEST AND RESPONSE MODELS
# ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    conversation_history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    citations: List[str]
    tools_used: List[str]
    query_type: str


class UploadResponse(BaseModel):
    """Response model for file upload endpoint"""
    success: bool
    filename: str
    message: str
    chunks_indexed: int = 0


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    agent_status: dict
    rag_ready: bool
    available_tools: List[str]


# ─────────────────────────────────────────────────────────
# FASTAPI APP INITIALIZATION
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title="JengaVest AI Backend",
    description="AI-powered financial analyst backend",
    version="0.1.0"
)

# Enable CORS for Next.js frontend
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.vercel.app",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
agent = None
rag_initialized = False


# ─────────────────────────────────────────────────────────
# STARTUP AND SHUTDOWN
# ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize agent and RAG pipeline on app startup"""
    global agent, rag_initialized
    
    print("\n" + "="*60)
    print("JENGAVEST AI BACKEND - STARTUP")
    print("="*60 + "\n")
    
    # Initialize agent
    try:
        agent = get_agent()
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        agent = None
    
    # Initialize RAG pipeline
    try:
        initialize_rag_pipeline()
        rag_initialized = True
        print("✓ RAG pipeline initialized")
    except Exception as e:
        print(f"⚠ Warning: RAG pipeline initialization failed: {e}")
        rag_initialized = False
    
    print("\n" + "="*60)
    print("Backend ready for requests")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────
# HEALTH CHECK ENDPOINT
# ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint that returns API status
    and environment inspection results.
    """
    if not agent:
        return HealthResponse(
            status="error",
            agent_status={"error": "Agent not initialized"},
            rag_ready=rag_initialized,
            available_tools=[tool["name"] for tool in TOOLS]
        )
    
    return HealthResponse(
        status="healthy",
        agent_status=agent.status,
        rag_ready=rag_initialized,
        available_tools=[tool["name"] for tool in TOOLS]
    )


# ─────────────────────────────────────────────────────────
# CHAT ENDPOINT
# ─────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for non-streaming responses.
    
    Request:
    - message: User's message
    - conversation_history: Previous messages (optional)
    
    Returns:
    - response: AI response with citations
    - citations: List of sources cited
    - tools_used: List of tools called
    - query_type: Type of query routed
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Run agent
        result = agent.run_agent(
            user_message=request.message,
            conversation_history=request.conversation_history
        )
        
        return ChatResponse(
            response=result["response"],
            citations=result["citations"],
            tools_used=result["tools_used"],
            query_type=result["route"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# ─────────────────────────────────────────────────────────
# STREAMING CHAT ENDPOINT
# ─────────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint that returns SSE stream.
    
    Request:
    - message: User's message
    - conversation_history: Previous messages (optional)
    
    Returns:
    - Server-Sent Events stream of response chunks
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    async def event_generator():
        """Generate SSE events from agent stream"""
        try:
            for event in agent.stream_agent(
                user_message=request.message,
                conversation_history=request.conversation_history
            ):
                yield event
            
            # Send final "done" event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ─────────────────────────────────────────────────────────
# FILE UPLOAD ENDPOINT
# ─────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be indexed by RAG pipeline.
    
    Accepts:
    - file: PDF file to upload
    
    Returns:
    - success: Whether upload succeeded
    - filename: Name of uploaded file
    - message: Status message
    - chunks_indexed: Number of chunks created from PDF
    """
    
    if not rag_initialized:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Save PDF to documents folder
        documents_folder = "./documents"
        os.makedirs(documents_folder, exist_ok=True)
        
        file_path = os.path.join(documents_folder, file.filename)
        
        # Save file
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Index the PDF with RAG pipeline
        from rag import ingest_documents, index_chunks_in_chromadb, bm25_index
        
        chunks = ingest_documents(documents_folder)
        
        if chunks:
            # Index in ChromaDB
            index_chunks_in_chromadb(chunks)
            
            # Update BM25 index
            bm25_index.create_index(chunks)
            
            chunks_indexed = len(chunks)
        else:
            chunks_indexed = 0
        
        return UploadResponse(
            success=True,
            filename=file.filename,
            message=f"Document uploaded and indexed successfully",
            chunks_indexed=chunks_indexed
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )


# ─────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "JengaVest AI Backend",
        "version": "0.1.0",
        "description": "AI-powered financial analyst",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "upload": "POST /upload"
        }
    }


# ─────────────────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
