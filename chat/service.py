from db import crud
from agentic_rag_core import RobustAyurvedicRAG

rag_system: RobustAyurvedicRAG | None = None


def init_rag(rag: RobustAyurvedicRAG):
    global rag_system
    rag_system = rag


def run_chat(db, user, query: str, conversation_id: str | None):

    if rag_system is None:
        raise RuntimeError("RAG system not initialized")

    # ------------------------------------------------------------------
    # 1. Create or validate conversation
    # ------------------------------------------------------------------
    if not conversation_id:
        conv = crud.create_conversation(db, user.id, title=query[:50])
        conversation_id = str(conv.id)
    else:
        conv = crud.get_conversation(db, conversation_id, user.id)
        if not conv:
            raise Exception("Conversation not found")

    # ------------------------------------------------------------------
    # 2. Store user message
    # ------------------------------------------------------------------
    crud.create_message(
        db=db,
        conversation_id=conversation_id,
        role="user",
        content=query
    )

    # ------------------------------------------------------------------
    # 3. Load DB history
    # ------------------------------------------------------------------
    history = crud.get_conversation_messages(db, conversation_id)

    formatted_history = [
        {"role": m.role, "content": m.content}
        for m in history[:-1][-8:]
    ]

    # ------------------------------------------------------------------
    # 4. Reset and replay memory into RAG
    # ------------------------------------------------------------------
    rag_system.reset_conversation(conversation_id)

    for msg in formatted_history:
        rag_system.chat(
            msg["content"],
            session_id=conversation_id,
            use_memory=True,
            verbose=False
        )

    # ------------------------------------------------------------------
    # 5. Ask RAG
    # ------------------------------------------------------------------
    result = rag_system.chat(
        query,
        session_id=conversation_id,
        use_memory=True,
        verbose=False
    )

    answer = result["answer"]
    confidence = result.get("confidence", 0.0)

    # ------------------------------------------------------------------
    # 6. Safety gate
    # ------------------------------------------------------------------
    if confidence < 0.45:
        answer = (
            "I’m not fully confident about this answer based on the available Ayurvedic texts. "
            "Could you please rephrase your question or provide more detail?"
        )
        result["sources"] = []

    # ------------------------------------------------------------------
    # 7. Store assistant message
    # ------------------------------------------------------------------
    crud.create_message(
        db=db,
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        confidence=str(confidence),
    )

    # ------------------------------------------------------------------
    # 8. Update user usage (FIXED)
    # ------------------------------------------------------------------

    tokens_used = 0

    # ✅ Case 1: Real token usage available from model / RAG
    if "usage" in result:
        usage = result["usage"]
        tokens_used = int(usage.get("total_tokens", 0))

    # ✅ Case 2: Partial token info
    elif "prompt_tokens" in result or "completion_tokens" in result:
        tokens_used = int(result.get("prompt_tokens", 0)) + int(result.get("completion_tokens", 0))

    # ✅ Case 3: Safe fallback (heuristic)
    else:
        tokens_used = max(1, len(query + answer) // 4)

    crud.increment_usage(
        db=db,
        user_id=user.id,
        tokens=tokens_used
    )

    # ------------------------------------------------------------------
    # 9. Response
    # ------------------------------------------------------------------
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "confidence": confidence,
        "sources": result.get("sources", []),
    }
