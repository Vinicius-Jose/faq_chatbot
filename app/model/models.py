from typing import Annotated
from pydantic import BaseModel, Field


Neo4jKey = Annotated[str, Field(json_schema_extra={"neo4j_key": True})]


class User(BaseModel):
    email: Neo4jKey
    username: str = Field(default="")
    full_name: str = Field(default="")
    password: str = Field(default="")


class Message(BaseModel):
    user_email: str = Field()
    session_id: str | None = Field(default=None)
    text: str = Field()


class LLMResponseEndpoint(BaseModel):
    answer: str = Field()
    session_id: str | None = Field(default=None)


class Sessions(BaseModel):
    sessions: list[str] = Field(default=[])


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None
