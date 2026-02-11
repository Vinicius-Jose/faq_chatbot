from app.database.database import Neo4jDatabase
from app.services.llm import LLM, EmbbeddingHuggingFace
from neo4j_graphrag.types import LLMMessage
from os import getenv
from tests import SESSION_ID
import json


def test_llm_invoke() -> None:
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    result = chat_model.invoke("When the second world war ended?")
    assert result
    assert isinstance(result.content, str)


def test_llm_ainvoke() -> None:
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    result = chat_model.ainvoke("When the second world war ended?")
    assert result


def test_llm_invoke_with_message_history() -> None:
    embedder = EmbbeddingHuggingFace()
    db = Neo4jDatabase(embedder)
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    history = db.get_message_history(session_id=SESSION_ID)
    message = LLMMessage(
        role="user", content="What is python? Explain with a maximum of 10 words"
    )
    first_response = chat_model.invoke(input=message["content"])

    history.add_message(message)
    history.add_message(LLMMessage(role="assistant", content=first_response.content))
    message = LLMMessage(
        role="user",
        content="Explain again with another words and using less words than the last response",
    )
    second_response = chat_model.invoke(
        input=message["content"], message_history=history.messages
    )
    history.add_message(message)
    history.add_message(LLMMessage(role="assistant", content=second_response.content))
    word_count_response_1 = len(first_response.content.split(" "))
    word_count_response_2 = len(second_response.content.split(" "))
    assert word_count_response_1 <= 10
    assert word_count_response_2 < word_count_response_1
    assert first_response.content.lower() != second_response.content.lower()
    history.clear(True)
