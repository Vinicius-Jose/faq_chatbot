from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi import status
from uuid import uuid4
from os import getenv

from app.controller.user import get_current_active_user
from app.database.database import Neo4jDatabase
from app.model.models import LLMResponseEndpoint, Sessions, User
from app.model.models import Message
from app.services.llm import LLM, EmbbeddingHuggingFace
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.generation.prompts import RagTemplate

from app.utils.prompts import DEFAULT_SYSTEM_INSTRUCTIONS

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/")
def post(
    message: Message,
    user: Annotated[User, Depends(get_current_active_user)],
) -> LLMResponseEndpoint:
    _, history, llm = initialize_llm(message, user)
    llm = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    llm_response = llm.invoke(message.text, message_history=history)
    message_llm = LLMMessage(role="user", content=message.text)
    response_llm = LLMMessage(role="assistant", content=llm_response.content)
    history.add_messages([message_llm, response_llm])
    return LLMResponseEndpoint(
        answer=llm_response.content, session_id=message.session_id
    )


@router.get("/sessions", tags=["session"])
def get_sessions(
    user: Annotated[User, Depends(get_current_active_user)],
) -> Sessions:
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    sessions = db.get_sessions_from_user(user)
    if sessions:
        sessions = sessions[0]
    return Sessions(sessions=sessions)


@router.get("/sessions/{session_id}", tags=["session"])
def get_message_from_session(
    user: Annotated[User, Depends(get_current_active_user)], session_id: str
) -> List[LLMMessage]:
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    if check_session_user(session_id, user, db=db):
        history = db.get_message_history(session_id)
        return history.messages
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail="Session not found",
    )


@router.delete("/sessions", tags=["session"])
def delete_session(
    session_id: str,
    user: Annotated[User, Depends(get_current_active_user)],
) -> Sessions:
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    if check_session_user(session_id, user, db=db):
        history = db.get_message_history(session_id=session_id)
        history.clear(True)
        return Sessions(sessions=[session_id])
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        detail="Session not found",
    )


def check_session_user(session_id: str, user: User, db: Neo4jDatabase) -> bool:
    sessions = db.get_sessions_from_user(user)
    if sessions:
        return any(1 for session in sessions[0] if session == session_id)
    return False


@router.post("/rag/", tags=["rag"])
def post(
    message: Message,
    user: Annotated[User, Depends(get_current_active_user)],
) -> LLMResponseEndpoint:
    db, history, llm = initialize_llm(message, user)
    rag_template = RagTemplate(system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS)
    response_llm = db.rag_response(llm, message.text, history, rag_template)
    message_llm = LLMMessage(role="user", content=message.text)
    response_llm = LLMMessage(role="assistant", content=response_llm.answer)
    history.add_messages([message_llm, response_llm])
    return LLMResponseEndpoint(
        answer=response_llm["content"], session_id=message.session_id
    )


def initialize_llm(message: Message, user: User):
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)

    if not check_session_user(message.session_id, user, db=db):
        message.session_id = str(uuid4())

    history = db.get_message_history(session_id=message.session_id)
    db.link_basemodel_to_session(user, message.session_id)
    llm = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    return db, history, llm
