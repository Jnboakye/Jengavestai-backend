# main.py - FastAPI application for JengaVest AI backend

import os
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import get_agent
from rag import initialize_rag_pipeline
from tools import TOOLS
from stocks import list_stocks, search_stocks, get_quote, get_quotes, get_history, search_symbols, get_news
import auth

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

# Enable CORS for the Next.js frontend (local + Vercel + Netlify previews).
# If you deploy the frontend on a custom domain, add it to the regex below.
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.(vercel\.app|netlify\.app)",
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
    
    # NOTE: this is a *sync* generator on purpose. stream_agent does blocking
    # network I/O to Claude; wrapping it in an async generator would block the
    # event loop and make Starlette buffer the whole response (no live stream).
    # A plain generator is run in a threadpool and flushed chunk-by-chunk.
    def event_generator():
        """Generate SSE events from the agent stream."""
        try:
            for event in agent.stream_agent(
                user_message=request.message,
                conversation_history=request.conversation_history,
            ):
                yield event
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
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
# STOCK ENDPOINTS (live market data via yfinance)
# ─────────────────────────────────────────────────────────

@app.get("/stocks")
async def stocks(q: Optional[str] = None):
    """
    Return the tradable stock catalogue, optionally filtered by `q`
    (matches ticker or company name). Prices are not included here —
    the frontend fetches live quotes separately.
    """
    return {"stocks": search_stocks(q) if q else list_stocks()}


@app.get("/stocks/quotes")
async def stocks_quotes(symbols: str):
    """
    Live quotes for several tickers at once (comma-separated `symbols`),
    used to show current prices in the Markets list.
    """
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return {"quotes": get_quotes(tickers)}


@app.get("/documents")
async def list_documents():
    """List the PDF documents currently indexed for RAG search."""
    folder = "./documents"
    if not os.path.isdir(folder):
        return {"documents": []}
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
    return {"documents": files}


@app.get("/news")
async def market_news(symbols: Optional[str] = None):
    """
    Market news. With `symbols` (comma-separated held tickers), returns news
    for those companies plus general market headlines; otherwise general only.
    """
    tickers = [s.strip().upper() for s in symbols.split(",")] if symbols else []
    tickers = [t for t in tickers if t]
    try:
        return {"articles": get_news(tickers)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch news: {e}")


@app.get("/stocks/search")
async def stocks_search(q: str):
    """
    Search the full symbol universe (any listed stock/ETF), not just the
    bundled catalogue.
    """
    if not q or not q.strip():
        return {"results": []}
    try:
        return {"results": search_symbols(q.strip())}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e}")


@app.get("/stocks/{ticker}/quote")
async def stock_quote(ticker: str):
    """Live quote (price + day change) for a single ticker."""
    try:
        return get_quote(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch quote for {ticker}: {e}")


@app.get("/stocks/{ticker}/history")
async def stock_history(ticker: str, range: str = "1M"):
    """
    Historical price series for a ticker.
    `range` is one of: 1D, 1W, 1M, 1Y, 5Y.
    """
    range_key = range.upper()
    if range_key not in {"1D", "1W", "1M", "1Y", "5Y"}:
        raise HTTPException(status_code=400, detail="range must be one of 1D, 1W, 1M, 1Y, 5Y")
    try:
        return get_history(ticker.upper(), range_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch history for {ticker}: {e}")


# ─────────────────────────────────────────────────────────
# AUTH (Supabase, backend-owned) + per-user portfolio
# ─────────────────────────────────────────────────────────

class MagicLinkRequest(BaseModel):
    email: str


class RefreshRequest(BaseModel):
    refresh_token: str


class AddHoldingRequest(BaseModel):
    ticker: str
    name: str
    sector: str
    amount_usd: float
    price: float


def require_user(authorization: Optional[str] = Header(None)) -> dict:
    """Validate the Bearer session token and return the user (or 401)."""
    if not auth.auth_enabled:
        raise HTTPException(status_code=401, detail="Auth is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    try:
        return auth.verify(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")


@app.get("/auth/status")
async def auth_status():
    """Tells the frontend whether real auth is available (else guest mode)."""
    return {"enabled": auth.auth_enabled}


@app.get("/auth/login/google")
async def auth_login_google():
    if not auth.auth_enabled:
        raise HTTPException(status_code=503, detail="Auth is not configured")
    return RedirectResponse(auth.start_google())


@app.post("/auth/magic-link")
async def auth_magic_link(req: MagicLinkRequest):
    if not auth.auth_enabled:
        raise HTTPException(status_code=503, detail="Auth is not configured")
    try:
        auth.start_magic_link(req.email)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not send magic link: {e}")


@app.get("/auth/callback")
async def auth_callback(code: Optional[str] = None, error: Optional[str] = None):
    """
    Where Google / magic link land. Exchange the code for a session, then
    redirect back to the frontend with the tokens in the URL fragment (so they
    aren't logged), where the frontend stores them for bearer auth.
    """
    if error or not code:
        return RedirectResponse(f"{auth.FRONTEND_URL}/login?error=auth_failed")
    try:
        tokens = auth.exchange_code(code)
        frag = f"at={tokens['access_token']}&rt={tokens['refresh_token']}"
        return RedirectResponse(f"{auth.FRONTEND_URL}/auth/callback#{frag}")
    except Exception:
        return RedirectResponse(f"{auth.FRONTEND_URL}/login?error=auth_failed")


@app.post("/auth/refresh")
async def auth_refresh(req: RefreshRequest):
    try:
        return auth.refresh(req.refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Could not refresh session")


@app.get("/auth/me")
async def auth_me(user: dict = Depends(require_user)):
    return user


@app.post("/auth/logout")
async def auth_logout():
    # Stateless: the frontend drops its tokens. (Access tokens are short-lived.)
    return {"ok": True}


def _row_to_holding(row: dict) -> dict:
    return {
        "ticker": row["ticker"],
        "name": row["name"],
        "sector": row["sector"],
        "amountUsd": float(row["amount_usd"]),
        "purchasePrice": float(row["purchase_price"]),
        "addedAt": row["added_at"],
    }


@app.get("/portfolio")
async def get_portfolio(user: dict = Depends(require_user)):
    res = auth.service().table("holdings").select("*").eq("user_id", user["id"]).order("added_at").execute()
    return {"holdings": [_row_to_holding(r) for r in (res.data or [])]}


@app.post("/portfolio")
async def add_portfolio(req: AddHoldingRequest, user: dict = Depends(require_user)):
    if req.amount_usd <= 0 or req.price <= 0:
        raise HTTPException(status_code=400, detail="amount and price must be positive")

    svc = auth.service()
    existing = (
        svc.table("holdings").select("*").eq("user_id", user["id"]).eq("ticker", req.ticker).execute()
    )
    rows = existing.data or []
    if rows:
        old = rows[0]
        old_amount = float(old["amount_usd"])
        old_purchase = float(old["purchase_price"])
        old_shares = old_amount / old_purchase
        new_shares = req.amount_usd / req.price
        total_amount = old_amount + req.amount_usd
        total_shares = old_shares + new_shares
        purchase_price = total_amount / total_shares
    else:
        total_amount = req.amount_usd
        purchase_price = req.price

    svc.table("holdings").upsert(
        {
            "user_id": user["id"],
            "ticker": req.ticker,
            "name": req.name,
            "sector": req.sector,
            "amount_usd": total_amount,
            "purchase_price": purchase_price,
        },
        on_conflict="user_id,ticker",
    ).execute()
    return {"ok": True}


@app.delete("/portfolio/{ticker}")
async def delete_holding(ticker: str, user: dict = Depends(require_user)):
    auth.service().table("holdings").delete().eq("user_id", user["id"]).eq("ticker", ticker.upper()).execute()
    return {"ok": True}


@app.delete("/portfolio")
async def reset_portfolio(user: dict = Depends(require_user)):
    auth.service().table("holdings").delete().eq("user_id", user["id"]).execute()
    return {"ok": True}


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
            "upload": "POST /upload",
            "stocks": "GET /stocks",
            "stock_quote": "GET /stocks/{ticker}/quote",
            "stock_history": "GET /stocks/{ticker}/history?range=1D|1W|1M|1Y|5Y"
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
