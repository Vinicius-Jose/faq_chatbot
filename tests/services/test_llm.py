from app.services.llm import LLM
from os import getenv
from dotenv import load_dotenv

load_dotenv("./.env")


def test_llm_invoke() -> None:
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    result = chat_model.invoke("When the second world war ended?")
    assert result is not None
    assert isinstance(result.content, str)


def test_llm_ainvoke() -> None:
    chat_model = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    result = chat_model.ainvoke("When the second world war ended?")
    assert result is not None
