from app.database.database import Neo4jDatabase
from app.model.models import User
from app.services.llm import LLM, EmbbeddingHuggingFace, DEFAULT_SYSTEM_INSTRUCTIONS
from langchain_text_splitters import TokenTextSplitter
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.generation.prompts import RagTemplate
from os import getenv
import pytest
from typing import Generator, Tuple
from time import sleep
from tests import PATH_PDF_SAMPLE, DOCUMENT_METADATA, SESSION_ID
import uuid


def setup() -> Tuple[
    Neo4jDatabase,
    LLM,
    EmbbeddingHuggingFace,
    TokenTextSplitter,
    LangChainTextSplitterAdapter,
]:
    db = Neo4jDatabase()
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    embedder = EmbbeddingHuggingFace()
    splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=10)
    adapter_splitter = LangChainTextSplitterAdapter(splitter)
    return db, chat_model, embedder, adapter_splitter


@pytest.fixture(scope="module")
def setup_pdf_sample() -> Generator[dict, None, None]:
    db, chat_model, embedder, adapter_splitter = setup()
    result = db.create_graph_from_pdf(
        llm=chat_model,
        embedder=embedder,
        file_path=PATH_PDF_SAMPLE,
        document_metada=DOCUMENT_METADATA,
        text_splitter=adapter_splitter,
    )
    data = {
        "db": db,
        "chat_model": chat_model,
        "embedder": embedder,
        "result": result,
    }
    yield data
    db.delete_document_with_metadata(metadata=DOCUMENT_METADATA)


def test_database_neo4j_connection() -> None:
    db = Neo4jDatabase()
    assert db.get_graph()._check_driver_state() is None


def test_insert_basemodel_user() -> None:
    db, _, _, _ = setup()
    user = User(
        email="teste@email.com",
        username="Teste",
        password="123",
        full_name="Teste Testado",
    )
    records = db.save_basemodel(user)
    assert len(records) == 1
    records = db.delete_basemodel(user)
    assert len(records) == 0


def test_get_basemodel_user() -> None:
    db, _, _, _ = setup()
    user = User(
        email="teste@email.com",
        username="Teste",
        password="123",
        full_name="Teste Testado",
    )
    db.save_basemodel(user)
    user_bd = db.get_basemodel(user)
    assert isinstance(user_bd, User)
    assert user_bd.email == user.email
    assert user_bd.password == user.password
    records = db.delete_basemodel(user)
    assert len(records) == 0


def test_create_graph_from_pdf(setup_pdf_sample: dict) -> None:
    result = setup_pdf_sample.get("result")
    assert result is not None


def test_rag_response(setup_pdf_sample: dict) -> None:
    sample_data = setup_pdf_sample
    db: Neo4jDatabase = sample_data.get("db")
    chat_model: LLM = sample_data.get("chat_model")
    chat_model_retriever = LLM(
        model_name=f"groq:{getenv('GROQ_MODEL')}",
        model_params={"model_kwargs": {"response_format": {"type": "text"}}},
    )
    db.set_retriever(chat_model_retriever)
    rag_template = RagTemplate(system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS)
    query = """How does providing a clear and precise prompt help an LLM understand the task and generate accurate, relevant responses?"""
    sleep(120)  # Since free tier has a low limit of token usage per minute
    response = db.rag_response(
        llm=chat_model, query_text=query, rag_template=rag_template
    )
    assert response.answer is not None


def test_get_message_history() -> None:
    db, _, _, _ = setup()
    history = db.get_message_history(session_id=SESSION_ID)
    assert len(history.messages) == 0
    history.clear(True)
    assert len(history.messages) == 0


def test_save_message_history() -> None:
    db, _, _, _ = setup()
    message = LLMMessage(role="user", content="Making a Test")
    history = db.get_message_history(session_id=SESSION_ID)
    history.add_message(message)
    history.add_message(LLMMessage(role="assistant", content="Test Answer"))
    assert len(history.messages) == 2
    history.clear(True)
    assert len(history.messages) == 0


def test_link_basemodel_to_session() -> None:
    db, _, _, _ = setup()
    user = User(
        email="teste@email.com",
        username="Teste",
        password="123",
        full_name="Teste Testado",
    )
    db = Neo4jDatabase()
    db.save_basemodel(user)
    session_id = str(uuid.uuid4())
    history = db.get_message_history(session_id)
    db.link_basemodel_to_session(user, session_id)
    result = db.get_sessions_from_user(user)
    assert result[0][0] == session_id
    history.clear(True)
