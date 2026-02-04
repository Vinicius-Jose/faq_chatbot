from app.database.database import Neo4jDatabase
from app.services.llm import LLM
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
    db = Neo4jDatabase()
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    history = db.get_message_history(session_id=SESSION_ID)
    message = LLMMessage(
        role="user", content="What is python? Explain with a maximum of 10 words"
    )
    first_response = chat_model.invoke(input=message["content"])

    history.add_message(message)
    first_response = json.loads(first_response.content)
    history.add_message(LLMMessage(role="assistant", content=first_response["answer"]))
    message = LLMMessage(
        role="user",
        content="Explain again with another words and with a maximum of 5 words",
    )
    second_response = chat_model.invoke(
        input=message["content"], message_history=history.messages
    )
    history.add_message(message)
    second_response = json.loads(second_response.content)
    history.add_message(LLMMessage(role="assistant", content=second_response["answer"]))
    word_count_response_1 = len(second_response["answer"].split(" "))
    word_count_response_2 = len(first_response["answer"].split(" "))
    assert word_count_response_1 <= 10
    assert word_count_response_2 <= 5
    assert first_response["answer"].lower() != second_response["answer"].lower()
    history.clear(True)
