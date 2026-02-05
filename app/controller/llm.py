from fastapi import APIRouter, HTTPException
from fastapi import status
from uuid import uuid4
from os import getenv

from app.database.database import Neo4jDatabase
from app.model.models import LLMResponseEndpoint, Sessions, User
from app.model.models import Message
from app.services.llm import LLM


router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/")
def post(message: Message) -> LLMResponseEndpoint:
    db = Neo4jDatabase()
    try:
        user = db.get_basemodel(User(email=message.user_email))
    except (IndexError, TypeError):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    session_id = message.session_id if message.session_id else str(uuid4())
    history = db.get_message_history(session_id=session_id)
    db.link_basemodel_to_session(user, session_id)
    llm = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")

    llm_response = llm.invoke(message.text, message_history=history)
    return LLMResponseEndpoint(answer=llm_response.content, session_id=session_id)


@router.get("/sessions")
def get_sessions(email: str) -> Sessions:
    db = Neo4jDatabase()
    sessions = db.get_sessions_from_user(User(email=email))
    return Sessions(sessions=sessions[0])


@router.delete("/sessions")
def delete_session(email: str, session_id: str) -> Sessions:
    db = Neo4jDatabase()
    sessions = db.get_sessions_from_user(User(email=email))
    for session in sessions[0]:
        if session == session_id:
            history = db.get_message_history(session_id=session_id)
            history.clear(True)
            return Sessions(sessions=[session_id])
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail="Session not found",
    )
