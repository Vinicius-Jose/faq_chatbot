from dotenv import load_dotenv
from app.database.database import Neo4jDatabase
from app.model.models import User
from app.services.llm import LLM, EmbbeddingHuggingFace, DEFAULT_SYSTEM_INSTRUCTIONS
from langchain_text_splitters import TokenTextSplitter
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.generation.prompts import RagTemplate
from os import getenv
import pytest
from typing import Generator
from time import sleep
from tests import PATH_PDF_SAMPLE

load_dotenv("./.env")

DOCUMENT_METADATA = {"subject": "Test"}


@pytest.fixture(scope="module")
def setup_pdf_sample() -> Generator[dict, None, None]:
    db = Neo4jDatabase()
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    embedder = EmbbeddingHuggingFace()
    splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=10)
    adapter_splitter = LangChainTextSplitterAdapter(splitter)
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
    user = User(
        email="teste@email.com",
        username="Teste",
        password="123",
        full_name="Teste Testado",
    )
    db = Neo4jDatabase()
    records = db.save_basemodel(user)
    assert len(records) == 1
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
