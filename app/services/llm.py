from os import getenv
from typing import Any, Coroutine, List, Sequence
from neo4j_graphrag.llm.types import LLMResponse, ToolCallResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import RateLimitHandler
from app.utils.tools import singleton
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from app.utils.prompts import DEFAULT_SYSTEM_INSTRUCTIONS


class LLM(LLMInterface):

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any] | None = {},
        rate_limit_handler: RateLimitHandler | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, model_params, rate_limit_handler, **kwargs)
        self.model = init_chat_model(
            self.model_name,
            temperature=model_params.get("temperature", 0.7),
            model_kwargs=model_params.get(
                "model_kwargs", {"response_format": {"type": "json_object"}}
            ),
        )

    def invoke(
        self,
        input: str,
        message_history: List[LLMMessage] | MessageHistory | None = [],
        system_instruction: str | None = DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> LLMResponse:
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=input),
        ]
        if message_history:
            messages = message_history + messages
        response = self.model.invoke(messages)
        return LLMResponse(content=response.content)

    def ainvoke(
        self,
        input: str,
        message_history: List[LLMMessage] | MessageHistory | None = [],
        system_instruction: str | None = DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> Coroutine[Any, Any, LLMResponse]:
        messages = message_history + [
            SystemMessage(content=system_instruction),
            HumanMessage(content=input),
        ]
        response = self.model.ainvoke(messages)
        return response

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: List[LLMMessage] | MessageHistory | None = [],
        system_instruction: str | None = DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> ToolCallResponse:
        agent = create_agent(
            self.model,
            tools=tools,
            system_prompt=system_instruction,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": input}]},
        )
        return result

    def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: List[LLMMessage] | MessageHistory | None = [],
        system_instruction: str | None = DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> Coroutine[Any, Any, ToolCallResponse]:
        agent = create_agent(
            self.model,
            tools=tools,
            system_prompt=system_instruction,
        )
        result = agent.ainvoke(
            {"messages": [{"role": "user", "content": input}]},
        )
        return result


class EmbbeddingHuggingFace(Embedder):
    def __init__(self, rate_limit_handler: RateLimitHandler | None = None):
        super().__init__(rate_limit_handler)
        self.embedder = HuggingFaceEndpointEmbeddings(
            model=getenv("HUGGINGFACE_EMBEDDER_MODEL"),
        )

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)
