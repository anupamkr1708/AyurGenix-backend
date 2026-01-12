"""
FastAPI Backend for Ayurvedic Agentic RAG (Groq API Version)
Optimized for Render free tier deployment
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, AsyncGenerator
import uuid
from datetime import datetime
import os
import asyncio
from dotenv import load_dotenv

from agentic_rag_core import RobustAyurvedicRAG

# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------

load_dotenv()

IS_PROD = os.getenv("ENV", "dev") == "prod"

LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
print("🧠 LangSmith enabled:", LANGSMITH_ENABLED)
if LANGSMITH_ENABLED:
    print("📊 LangSmith Project:", os.getenv("LANGCHAIN_PROJECT", "Not set"))

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ayurveda-rag-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    print("⚠️  Warning: PINECONE_API_KEY not set in environment")

if not GROQ_API_KEY:
    print("⚠️  Warning: GROQ_API_KEY not set in environment")

print("✅ Configuration loaded")
print(f"   Pinecone Index: {PINECONE_INDEX}")
print(f"   Groq API: {'Connected' if GROQ_API_KEY else 'Not configured'}")

# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="🌿 Ayurvedic Agentic RAG API (Groq)",
    description="""
Production-grade Agentic RAG system for Ayurveda powered by Groq API.

Features:
- ✅ Cloud-based LLM (Groq LLaMA 3.3 70B)
- ✅ Conversational Memory
- ✅ Agentic Retrieval & Reranking
- ✅ Streaming (SSE)
- ✅ LangSmith Tracing
- ✅ Confidence Scoring
- ✅ Optimized for Render Free Tier
""",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not IS_PROD else ["https://yourdomain.com"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------

rag: Optional[RobustAyurvedicRAG] = None
sessions: Dict[str, Dict] = {}

# -----------------------------------------------------------------------------
# STARTUP / SHUTDOWN
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system once at startup"""
    global rag

    print("\n" + "=" * 70)
    print("🚀 INITIALIZING AYURVEDIC AGENTIC RAG (GROQ)")
    print("=" * 70)

    # Check required environment variables
    if not PINECONE_API_KEY:
        print("❌ ERROR: PINECONE_API_KEY not found in environment variables")
        print("   Please set it in your .env file")
        return
    
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not found in environment variables")
        print("   Please set it in your .env file")
        return

    try:
        rag = RobustAyurvedicRAG(
            pinecone_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX,
            groq_api_key=GROQ_API_KEY,
        )
        print("✅ RAG system ready")
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        print("   Check your API keys and network connection")
        # Don't raise - allow server to start for health checks

    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 API shutting down gracefully")

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

def get_rag() -> RobustAyurvedicRAG:
    """Dependency to ensure RAG is initialized"""
    if rag is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not initialized. Check server logs for API key issues."
        )
    return rag

# -----------------------------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(
        ..., 
        example="What are the symptoms of pitta imbalance?",
        description="User's question about Ayurveda"
    )
    session_id: Optional[str] = Field(
        None, 
        description="Client session ID (UUID). If not provided, a new session is created."
    )
    use_memory: bool = Field(
        True, 
        description="Enable conversational memory for context-aware responses"
    )


class Source(BaseModel):
    source: str
    page: int
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: List[Source]
    confidence: float
    reasoning: List[str]
    intent: str
    entities: List[str]
    response_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_sessions: int
    pinecone_index: str
    groq_model: str
    langsmith_enabled: bool


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "🌿 Ayurvedic Agentic RAG",
        "status": "online",
        "version": "4.0.0",
        "llm_provider": "Groq (LLaMA 3.3 70B)",
        "deployment": "Render Free Tier Optimized",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "chat": "/chat",
            "stream": "/chat/stream",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if rag is not None else "initializing",
        model_loaded=rag is not None,
        active_sessions=len(sessions),
        pinecone_index=PINECONE_INDEX,
        groq_model="llama-3.3-70b-versatile",
        langsmith_enabled=LANGSMITH_ENABLED,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RobustAyurvedicRAG = Depends(get_rag),
):
    """
    Non-streaming chat endpoint
    
    Returns complete response with sources, confidence, and reasoning steps.
    """

    # Create or retrieve session
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
        }

    try:
        # Run RAG in thread pool to avoid blocking
        result = await asyncio.to_thread(
            rag.chat,
            request.query,
            session_id=session_id,
            use_memory=request.use_memory,
            verbose=False,
        )

        # Update session stats
        sessions[session_id]["message_count"] += 1
        sessions[session_id]["last_active"] = datetime.now().isoformat()

        # Convert to response model
        return ChatResponse(
            session_id=session_id,
            query=result["query"],
            answer=result["answer"],
            sources=[
                Source(
                    source=s["source"],
                    page=s["page"],
                    score=s["score"],
                    text_preview=s["text_preview"],
                )
                for s in result["sources"]
            ],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            intent=result["intent"],
            entities=result["entities"],
            response_time_seconds=result["response_time_seconds"],
        )

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    rag: RobustAyurvedicRAG = Depends(get_rag),
):
    """
    Streaming SSE endpoint (token-by-token)
    
    Returns Server-Sent Events with streaming tokens.
    """

    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Get conversation context
            memory = rag._get_memory(session_id)
            conversation_context = ""
            if request.use_memory and memory.messages:
                conversation_context = memory.get_relevant(request.query, n=3)
            
            # Process query
            query_info = rag.query_processor.process(request.query, conversation_context)
            
            # Retrieve and rerank
            raw_contexts = rag.retrieve(query_info["expanded_queries"], top_k=5)
            reranked_contexts = rag.reranker.rerank(
                request.query, raw_contexts, intent=query_info["intent"], top_k=5
            )
            
            # Build prompt
            context_text = "\n\n".join(
                [
                    f"[Source {i+1}: {c['source']}, Page {c['page']}]\n{c['text']}"
                    for i, c in enumerate(reranked_contexts)
                ]
            )
            
            intent_prompts = {
                "treatment": "You are an Ayurvedic physician. Provide treatment recommendations.",
                "diet": "You are an Ayurvedic nutritionist. Provide dietary guidance.",
                "lifestyle": "You are an Ayurvedic lifestyle consultant.",
                "general": "You are an Ayurvedic scholar.",
            }
            
            system_prompt = intent_prompts.get(query_info["intent"], intent_prompts["general"])
            
            prompt = f"""Answer based on these Ayurvedic texts:

CONTEXT:
{context_text}

QUESTION: {request.query}

Provide a comprehensive answer with practical guidance.

ANSWER:"""
            
            # Stream generation
            for token in rag.generate_stream(prompt=prompt, system_prompt=system_prompt):
                yield f"data: {token}\n\n"
                await asyncio.sleep(0)

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"❌ Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    rag: RobustAyurvedicRAG = Depends(get_rag),
):
    """
    Delete a session and clear its memory
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    rag.reset_conversation(session_id)
    del sessions[session_id]

    return {
        "status": "deleted",
        "session_id": session_id,
        "message": "Session memory cleared successfully"
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "created_at": data["created_at"],
                "message_count": data["message_count"],
                "last_active": data.get("last_active", data["created_at"]),
            }
            for sid, data in sessions.items()
        ],
    }


@app.get("/stats")
async def stats():
    """Get system statistics"""
    total_messages = sum(s["message_count"] for s in sessions.values())
    
    return {
        "total_sessions": len(sessions),
        "total_messages": total_messages,
        "avg_messages_per_session": (
            round(total_messages / len(sessions), 2) if sessions else 0
        ),
        "llm_provider": "Groq",
        "model": "llama-3.3-70b-versatile",
    }


@app.post("/clear-all-sessions")
async def clear_all_sessions(
    rag: RobustAyurvedicRAG = Depends(get_rag),
):
    """Clear all sessions (admin endpoint)"""
    rag.reset_conversation()
    sessions.clear()
    
    return {
        "status": "cleared",
        "message": "All sessions cleared successfully"
    }

# -----------------------------------------------------------------------------
# LOCAL DEVELOPMENT RUN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n🌐 Starting FastAPI server...")
    print("📘 API Docs → http://localhost:8000/docs")
    print("🔍 Health Check → http://localhost:8000/health")
    print("=" * 70)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )