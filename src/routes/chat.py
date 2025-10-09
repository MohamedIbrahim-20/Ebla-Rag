from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Optional

from helpers.config import get_settings, Settings
from helpers.prompts import build_prompt
from controllers import GroqRAGController
from models.db import SessionLocal
from models.schemas import Session as DBSess, Message as DBMsg, Retrieval as DBRet, Summary as DBSum
from .schemes.chat import ChatRequest

chat_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1", "chat"],
)

SUMMARY_THRESHOLD_MESSAGES = 5  # when total messages exceed this, attempt summarization
SUMMARY_BATCH_SIZE = 4          # how many oldest messages to compress per pass
RECENT_KEEP = 4                  # keep this many most-recent messages verbatim


def _get_or_create_session(db, system_prompt: Optional[str]) -> str:
    sess = DBSess(system_prompt=system_prompt)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess.id


def _get_latest_summary(db, session_id: str) -> Optional[str]:
    q = db.query(DBSum).filter(DBSum.session_id == session_id).order_by(DBSum.created_at.desc()).first()
    return q.summary if q else None


def _get_recent_messages(db, session_id: str, limit: int = 8) -> List[dict]:
    msgs = (
        db.query(DBMsg)
        .filter(DBMsg.session_id == session_id)
        .order_by(DBMsg.created_at.desc())
        .limit(limit)
        .all()
    )
    # reverse to chronological
    msgs = list(reversed(msgs))
    return [{"role": m.role, "content": m.content} for m in msgs]


def _maybe_summarize(db, session_id: str, rag: GroqRAGController) -> None:
    """Create/append a rolling summary when history grows.
    Strategy:
      - Keep the last RECENT_KEEP messages verbatim
      - Summarize a batch of older messages (oldest first) when total > threshold
    """
    # total messages
    total_msgs = db.query(DBMsg).filter(DBMsg.session_id == session_id).count()
    if total_msgs <= SUMMARY_THRESHOLD_MESSAGES:
        return

    # find latest summary cutoff
    latest_sum: Optional[DBSum] = (
        db.query(DBSum)
        .filter(DBSum.session_id == session_id)
        .order_by(DBSum.created_at.desc())
        .first()
    )

    # messages ordered asc
    all_msgs = (
        db.query(DBMsg)
        .filter(DBMsg.session_id == session_id)
        .order_by(DBMsg.created_at.asc())
        .all()
    )

    if not all_msgs:
        return

    # Determine start index after last summarized message
    start_idx = 0
    if latest_sum:
        # find index of up_to_message_id
        for i, m in enumerate(all_msgs):
            if m.id == latest_sum.up_to_message_id:
                start_idx = i + 1
                break

    # Leave most recent RECENT_KEEP messages unsummarized
    cutoff_idx = max(0, len(all_msgs) - RECENT_KEEP)
    # Select the next batch to summarize
    batch = all_msgs[start_idx: min(start_idx + SUMMARY_BATCH_SIZE, cutoff_idx)]
    if not batch:
        return

    # Build plain text of the batch
    convo_text = []
    for m in batch:
        role = m.role.capitalize()
        convo_text.append(f"{role}: {m.content}")
    convo_str = "\n".join(convo_text)

    # Summarization prompt (concise, factual)
    sum_prompt = (
        "Summarize the following conversation turns concisely and factually so it can replace them in future context.\n"
        "Do not add new information; preserve key facts and decisions.\n\n" + convo_str + "\n\nSummary:"
    )

    try:
        if rag.llm is None:
            # ensure models
            if not rag.initialize_models():
                return
        resp = rag.llm.invoke(sum_prompt)
        summary_text = getattr(resp, "content", None)
        if not summary_text:
            return

        up_to_msg = batch[-1]
        new_sum = DBSum(session_id=session_id, up_to_message_id=up_to_msg.id, summary=summary_text)
        db.add(new_sum)
        db.commit()
    except Exception:
        db.rollback()
        # soft-fail summarization
        return


@chat_router.post("/chat")
def chat(req: ChatRequest, app_settings: Settings = Depends(get_settings)):
    if not req.message or not req.message.strip():
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "message is required"})
    method = req.method or "langchain"
    if method not in ("langchain", "llamaindex"):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "invalid method"})

    db = SessionLocal()
    try:
        # session resolution
        session_id = req.session_id
        if not session_id:
            session_id = _get_or_create_session(db, system_prompt=None)

        # persist user turn
        user_msg = DBMsg(session_id=session_id, role="user", content=req.message)
        db.add(user_msg)
        db.commit()

        # build context pieces
        latest_summary = _get_latest_summary(db, session_id)
        recent_messages = _get_recent_messages(db, session_id)

        # RAG retrieval
        rag = GroqRAGController(project_id="2", settings=app_settings)
        if not rag.initialize_models():
            return JSONResponse(status_code=500, content={"error": "Failed to init models"})
        results = rag.search_documents(req.message, method=method)
        retrieved_chunks = [r.get("content", "") for r in results]

        # prompt build
        # load session for system_prompt
        session_row = db.query(DBSess).filter(DBSess.id == session_id).first()
        system_prompt = session_row.system_prompt if session_row and session_row.system_prompt else None
        prompt = build_prompt(system_prompt, latest_summary, recent_messages, retrieved_chunks, req.message)

        # generate answer (using LangChain path via GroqRAGController.ask_question_langchain)
        # we already have retrieved docs; reuse RAG generation to keep behavior consistent
        if method == "langchain":
            answer_payload = rag.ask_question_langchain(req.message)
        else:
            answer_payload = rag.ask_question_llamaindex(req.message)
        if "error" in answer_payload:
            return JSONResponse(status_code=500, content=answer_payload)

        answer = answer_payload.get("answer", "")

        # persist assistant turn
        assistant_msg = DBMsg(session_id=session_id, role="assistant", content=answer)
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        # persist retrievals provenance for this assistant message
        to_insert = []
        for r in results:
            to_insert.append(DBRet(
                message_id=assistant_msg.id,
                source_type=method,
                source_id=str(r.get("document", {}).get("metadata", {}).get("id", "")),
                score=float(r.get("score", 0.0)) if r.get("score") is not None else None,
                chunk=r.get("content", ""),
                retrieval_metadata=r.get("document", {}).get("metadata", {}),
            ))
        if to_insert:
            db.add_all(to_insert)
            db.commit()

        # summarization pass (rolling)
        _maybe_summarize(db, session_id, rag)

        return JSONResponse(status_code=200, content={
            "session_id": session_id,
            "answer": answer,
            "retrieved_documents": results,
        })

    except Exception as e:
        db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        db.close()


@chat_router.get("/history/{session_id}")
def history(session_id: str, app_settings: Settings = Depends(get_settings)):
    db = SessionLocal()
    try:
        sess = db.query(DBSess).filter(DBSess.id == session_id).first()
        if not sess:
            return JSONResponse(status_code=404, content={"error": "session not found"})

        latest_summary = _get_latest_summary(db, session_id)
        msgs = (
            db.query(DBMsg)
            .filter(DBMsg.session_id == session_id)
            .order_by(DBMsg.created_at.asc())
            .all()
        )
        payload = {
            "session_id": session_id,
            "summary": latest_summary,
            "messages": [{"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in msgs]
        }
        return JSONResponse(status_code=200, content=payload)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        db.close()


