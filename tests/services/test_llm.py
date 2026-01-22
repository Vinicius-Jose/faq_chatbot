from app.services.llm import LLM, EmbbeddingHuggingFace


def test_llm_invoke() -> None:
    chat_model = LLM(model_name="groq:llama-3.3-70b-versatile")
    result = chat_model.invoke("When the second world war ended?")
    assert result is not None


def test_llm_ainvoke() -> None:
    chat_model = LLM(model_name="groq:llama-3.3-70b-versatile")
    result = chat_model.ainvoke("When the second world war ended?")
    assert result is not None
