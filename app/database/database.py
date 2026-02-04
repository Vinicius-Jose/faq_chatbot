from typing import Any
from langchain_neo4j import Neo4jGraph
from os import getenv

from neo4j import EagerResult
from neo4j.exceptions import ConstraintError
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from app.utils.tools import singleton
from pydantic import BaseModel
import asyncio
from neo4j_graphrag.retrievers.vector import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter

from neo4j_graphrag.generation.prompts import RagTemplate

from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.message_history import Neo4jMessageHistory


@singleton
class Neo4jDatabase:
    def __init__(self, embedder: Embedder) -> None:
        url: str = getenv("NEO4J_URI")
        username: str = getenv("NEO4J_USERNAME")
        password: str = getenv("NEO4J_PASSWORD")
        self.database: str = getenv("NEO4J_DATABASE")
        self.__graph: Neo4jGraph = Neo4jGraph(
            url=url, username=username, password=password, database=self.database
        )
        self.__driver = self.__graph._driver
        self.embedder = embedder
        self.retriever = None
        self.vector_dimensions = int(getenv("VECTOR_DIMENSIONS"))

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

    def get_basemodel(self, model: BaseModel) -> BaseModel:
        label, _, merge_keys, _, merge_data = self.__extract_keys_basemodel(model)

        query = f"""
        MATCH (n:{label} {{{merge_keys}}})
        RETURN n"""
        records, _, _ = self.__driver.execute_query(query, merge_data)
        model_cls = model.__class__
        model_found = model_cls(**records[0][0])
        return model_found

    def delete_basemodel(self, model: BaseModel) -> EagerResult:
        label, _, merge_keys, _, merge_data = self.__extract_keys_basemodel(model)
        query = f"""
            MATCH (n:{label} {{{merge_keys}}})
            DETACH DELETE n
        """
        records, _, _ = self.__driver.execute_query(query, merge_data)
        return records

    def create_graph_from_pdf(
        self,
        llm: LLMInterface,
        file_path: str,
        document_metada: dict = None,
        text_splitter: TextSplitter = None,
    ) -> PipelineResult:

        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=self.__driver,
            neo4j_database=self.database,
            embedder=self.embedder,
            from_pdf=True,
            text_splitter=text_splitter,
        )
        result = asyncio.run(
            kg_builder.run_async(file_path=file_path, document_metadata=document_metada)
        )
        self.__create_vector_index()
        return result

    def __create_vector_index(self) -> None:
        query = """
        CREATE VECTOR INDEX chunkEmbeddings IF NOT EXISTS 
        FOR (c:Chunk)
        ON c.embedding
        OPTIONS {
        indexConfig: {
            `vector.dimensions`: $dimensions,
            `vector.similarity_function`: 'cosine'
        }
        }"""
        self.__driver.execute_query(query, {"dimensions": self.vector_dimensions})

    def delete_document_with_metadata(self, metadata: dict) -> EagerResult:
        keys = "AND ".join([f"d.{k}= ${k}" for k in metadata.keys()])
        query = f"""
            MATCH (d:Document)-[r]-(c)-[re]-(e)
            WHERE {keys}
            DETACH DELETE d,c,e,r,re
        """
        records, _, _ = self.__driver.execute_query(query, metadata)
        return records

    def rag_response(
        self,
        llm: LLMInterface,
        query_text: str,
        message_history: list = [],
        rag_template: RagTemplate = None,
    ) -> RagResultModel:
        if not self.retriever:
            self.set_retriever(self.embedder)
        rag = GraphRAG(retriever=self.retriever, llm=llm, prompt_template=rag_template)
        response = rag.search(
            query_text=query_text, return_context=True, message_history=message_history
        )
        return response

    def set_retriever(self, embedder: Embedder):
        self.retriever = VectorRetriever(
            driver=self.__driver,
            embedder=embedder,
            index_name="chunkEmbeddings",
            neo4j_database=self.database,
            return_properties=["text"],
        )

    def get_message_history(
        self, session_id: str, window: str = 3
    ) -> Neo4jMessageHistory:
        history = Neo4jMessageHistory(
            session_id=session_id,
            driver=self.__driver,
            database=self.database,
            window=window,
        )
        return history

    def link_basemodel_to_session(
        self, model: BaseModel, session_id: str
    ) -> EagerResult:
        label, _, merge_keys, _, merge_data = self.__extract_keys_basemodel(model)

        query = f"""
        MATCH (n:{label} {{{merge_keys}}})
        MATCH (s:Session {{id: $session_id}})
        MERGE (n)-[r:HAS_CONVERSATION]->(s)
        RETURN n,r,s
        """
        merge_data["session_id"] = session_id
        records, _, _ = self.__driver.execute_query(query, merge_data)
        return records

    def get_sessions_from_user(self, user: BaseModel) -> EagerResult:
        label, _, merge_keys, _, merge_data = self.__extract_keys_basemodel(user)
        query = f"""
            MATCH (n:{label} {{{merge_keys}}})-[]->(s:Session)
            RETURN s.id
            """
        records, _, _ = self.__driver.execute_query(query, merge_data)
        return records
