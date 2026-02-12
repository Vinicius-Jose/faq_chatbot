from app.database.database import Neo4jDatabase
from app.model.models import User
from app.services.llm import LLM, EmbbeddingHuggingFace
from app.utils.prompts import DEFAULT_SYSTEM_INSTRUCTIONS
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
from tests import PATH_PDF_SAMPLE, DOCUMENT_METADATA, SESSION_ID, USER
import uuid


def setup() -> Tuple[
    Neo4jDatabase,
    LLM,
    EmbbeddingHuggingFace,
    TokenTextSplitter,
    LangChainTextSplitterAdapter,
]:

    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=10)
    adapter_splitter = LangChainTextSplitterAdapter(splitter)
    return db, chat_model, embedder, adapter_splitter


@pytest.fixture(scope="module")
def setup_pdf_sample() -> Generator[dict, None, None]:
    db, chat_model, embedder, adapter_splitter = setup()
    result = db.create_graph_from_pdf(
        llm=chat_model,
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
    db, _, _, _ = setup()
    assert db.get_graph()._check_driver_state() is None


def test_insert_basemodel_user() -> None:
    db, _, _, _ = setup()

    records = db.save_basemodel(USER)
    assert len(records) == 1
    records = db.delete_basemodel(USER)
    assert len(records) == 0


def test_get_basemodel_user() -> None:
    db, _, _, _ = setup()
    db.save_basemodel(USER)
    user_bd = db.get_basemodel(USER)
    assert isinstance(user_bd, User)
    assert user_bd.email == USER.email
    assert user_bd.password == USER.password
    records = db.delete_basemodel(USER)
    assert len(records) == 0


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
    db = Neo4jDatabase()
    db.save_basemodel(USER)
    session_id = str(uuid.uuid4())
    history = db.get_message_history(session_id)
    db.link_basemodel_to_session(USER, session_id)
    result = db.get_sessions_from_user(USER)
    assert result[0][0] == session_id
    history.clear(True)
    db.delete_basemodel(USER)


def test_create_graph_from_pdf(setup_pdf_sample: dict) -> None:
    result = setup_pdf_sample.get("result")
    assert result is not None


def test_retriever(setup_pdf_sample: dict) -> None:
    db: Neo4jDatabase = setup_pdf_sample.get("db")
    db.set_retriever(setup_pdf_sample.get("embedder"))
    retriever = db.retriever
    records = retriever.get_search_results(
        query_text="What is Large Language Models (LLM) ?"
    )
    assert len(records.records) > 1


def test_rag_response(setup_pdf_sample: dict) -> None:
    db: Neo4jDatabase = setup_pdf_sample.get("db")
    chat_model: LLM = setup_pdf_sample.get("chat_model")
    rag_template = RagTemplate(system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS)
    query = """How does providing a clear and precise prompt help an LLM understand the task and generate accurate, relevant responses?"""
    sleep(120)  # Since free tier has a low limit of token usage per minute
    response = db.rag_response(
        llm=chat_model, query_text=query, rag_template=rag_template
    )
    assert response.answer


def test_rag_response_with_message_history(setup_pdf_sample: dict) -> None:
    db: Neo4jDatabase = setup_pdf_sample.get("db")
    chat_model: LLM = setup_pdf_sample.get("chat_model")
    rag_template = RagTemplate(system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS)
    query = "How does providing a clear and precise prompt help an LLM understand the task and generate accurate, relevant responses?"
    message = LLMMessage(role="user", content=query)
    history = db.get_message_history(session_id=SESSION_ID)
    history.add_message(message)
    sleep(120)  # Since free tier has a low limit of token usage per minute
    response_question_1 = db.rag_response(
        llm=chat_model, query_text=query, rag_template=rag_template
    )
    message = LLMMessage(role="assistant", content=response_question_1.answer)
    history.add_message(message)
    query = "Explain again with some notes and other words"
    message = LLMMessage(role="user", content=query)
    history.add_message(message)
    response_question_2 = db.rag_response(
        llm=chat_model,
        query_text=query,
        rag_template=rag_template,
        message_history=history,
    )
    message = LLMMessage(role="assistant", content=response_question_2.answer)
    history.add_message(message)
    assert response_question_2.answer
    assert response_question_2.answer != response_question_1.answer
    history.clear(True)
