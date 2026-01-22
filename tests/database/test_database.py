from dotenv import load_dotenv
from app.database.database import Neo4jDatabase
from app.model.models import User
from app.services.llm import LLM, EmbbeddingHuggingFace
from os import getenv

load_dotenv("./.env")


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


def test_create_graph_from_pdf() -> None:
    db = Neo4jDatabase()
    url_file = "./tests/database/sample.pdf"

    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    embedder = EmbbeddingHuggingFace()
    result = db.create_graph_from_pdf(
        llm=chat_model, embedder=embedder, file_path=url_file
    )
    assert result is not None
