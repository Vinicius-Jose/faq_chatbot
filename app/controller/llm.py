import json
from fastapi import APIRouter
from app.model.models import LLMResponseEndpoint
from os import getenv
from app.model.models import Message
from app.services.llm import LLM

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/")
def post(message: Message) -> LLMResponseEndpoint:
    llm = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    llm_response = llm.invoke(message.text)
    return LLMResponseEndpoint(answer=llm_response.content)
