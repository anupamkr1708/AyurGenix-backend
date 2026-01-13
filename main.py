"""
FastAPI Backend for Ayurvedic Agentic RAG (Groq API)
Phase B – Platform backend completed
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from auth.routes import router as auth_router
from chat.routes import router as chat_router
from chat.stream import router as chat_stream_router
from usage.routes import router as usage_router
from admin.routes import router as admin_router

from core.middleware import request_logger
from core.exceptions import global_exception_handler

from agentic_rag_core import RobustAyurvedicRAG
from chat.service import init_rag

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------

load_dotenv()

IS_PROD = os.getenv("ENV", "dev") == "prod"
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ayurveda-rag-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("🧠 LangSmith enabled:", LANGSMITH_ENABLED)
print("✅ Configuration loaded")
print("   Pinecone Index:", PINECONE_INDEX)
print("   Groq API:", "Connected" if GROQ_API_KEY else "Not configured")

# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="🌿 Ayurvedic Agentic RAG API",
    description="Production-grade Agentic RAG backend for Ayurveda (Groq + Pinecone)",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not IS_PROD else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# middleware + exceptions
app.middleware("http")(request_logger)
app.add_exception_handler(Exception, global_exception_handler)

# routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(chat_stream_router)
app.include_router(usage_router)
app.include_router(admin_router)

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------

rag: RobustAyurvedicRAG | None = None

# -----------------------------------------------------------------------------
# STARTUP / SHUTDOWN
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global rag

    print("\n" + "=" * 70)
    print("🚀 INITIALIZING AYURVEDIC AGENTIC RAG (GROQ)")
    print("=" * 70)

    if not PINECONE_API_KEY or not GROQ_API_KEY:
        print("❌ Missing API keys. Check .env file.")
        return

    try:
        rag = RobustAyurvedicRAG(
            pinecone_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX,
            groq_api_key=GROQ_API_KEY,
        )

        init_rag(rag)
        print("✅ RAG system ready and injected")

    except Exception as e:
        print("❌ Failed to initialize RAG:", e)

    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 API shutting down gracefully")

# -----------------------------------------------------------------------------
# CORE ROUTES
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "🌿 Ayurvedic Agentic RAG",
        "status": "online",
        "phase": "Phase B – Platform backend complete",
        "llm": "Groq llama-3.3-70b-versatile",
        "features": [
            "JWT auth",
            "Conversation memory",
            "Agentic RAG",
            "Usage tracking",
            "Confidence gating",
            "Streaming",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if rag else "initializing",
        "rag_loaded": rag is not None,
        "pinecone_index": PINECONE_INDEX,
        "langsmith_enabled": LANGSMITH_ENABLED,
    }


from auth.security import get_current_user

@app.get("/me")
def read_me(user=Depends(get_current_user)):
    return {"id": str(user.id), "email": user.email}


@app.get("/stats")
def stats():
    return {
        "llm_provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "rag_loaded": rag is not None,
        "phase": "Phase B completed",
    }

# -----------------------------------------------------------------------------
# LOCAL DEV
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n🌐 Starting FastAPI server...")
    print("📘 Docs → http://localhost:8000/docs")
    print("❤️  Health → http://localhost:8000/health")
    print("=" * 70)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
