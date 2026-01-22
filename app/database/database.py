from typing import Any
from langchain_neo4j import Neo4jGraph
from os import getenv

from neo4j import EagerResult
from neo4j.exceptions import ConstraintError
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from app.utils.tools import singleton
from pydantic import BaseModel
import asyncio


@singleton
class Neo4jDatabase:
    def __init__(self) -> None:
        url: str = getenv("NEO4J_URI")
        username: str = getenv("NEO4J_USERNAME")
        password: str = getenv("NEO4J_PASSWORD")
        self.database: str = getenv("NEO4J_DATABASE")
        self.__graph: Neo4jGraph = Neo4jGraph(
            url=url, username=username, password=password, database=self.database
        )
        self.__driver = self.__graph._driver

    def get_graph(self) -> Neo4jGraph:
        return self.__graph

    def __extract_keys_basemodel(self, model: BaseModel) -> tuple:
        model_cls = model.__class__
        label = model_cls.__name__
        data = model.model_dump(exclude_none=True)
        keys = []
        merge_data = {}
        for name, field in model_cls.model_fields.items():
            if field.json_schema_extra and field.json_schema_extra.get("neo4j_key"):
                keys.append(name)
                merge_data[name] = data[name]

        if not keys:
            raise ConstraintError("No key provided")

        merge_keys = ", ".join([f"{k}: ${k}" for k in keys])
        set_props = ", ".join([f"n.{k} = ${k}" for k in data.keys() if k not in keys])
        return label, data, merge_keys, set_props, merge_data

    def save_basemodel(self, model: BaseModel) -> EagerResult:
        label, data, merge_keys, set_props, _ = self.__extract_keys_basemodel(model)

        query = f"""
        MERGE (n:{label} {{{merge_keys}}})
        SET {set_props}
        RETURN n
        """
        records, _, _ = self.__driver.execute_query(query, data)
        return records

    def delete_basemodel(self, model: BaseModel) -> EagerResult:
        label, _, merge_keys, _, merge_data = self.__extract_keys_basemodel(model)
        query = f"""
            MATCH (n:{label} {{{merge_keys}}})
            DETACH DELETE n
        """
        records, _, _ = self.__driver.execute_query(query, merge_data)
        return records

    def create_graph_from_pdf(self, llm, embedder, file_path) -> Any | dict:

        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=self.__driver,
            neo4j_database=self.database,
            embedder=embedder,
            from_pdf=True,
        )
        result = asyncio.run(kg_builder.run_async(file_path=file_path))
        return result.result

    def delete_with_label(self, label: list) -> EagerResult:
        query = f"""
            MATCH (n:{"|".joint(label)})
            DETACH DELETE n
        """
        records, _, _ = self.__driver.execute_query(query)
        return records
