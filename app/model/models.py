from typing import Annotated
from pydantic import BaseModel, Field


Neo4jKey = Annotated[str, Field(json_schema_extra={"neo4j_key": True})]


class User(BaseModel):
    email: Neo4jKey
    username: str = Field()
    full_name: str = Field()
    password: str = Field()
