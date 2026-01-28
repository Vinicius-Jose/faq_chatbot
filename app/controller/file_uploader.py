from fastapi import APIRouter, File, Form, UploadFile
from langchain_text_splitters import TokenTextSplitter
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from os import getenv
from pathlib import Path
import tempfile
from fastapi.exceptions import HTTPException
from fastapi import status
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

from app.database.database import Neo4jDatabase
from app.services.llm import LLM, EmbbeddingHuggingFace


router = APIRouter(prefix="/files", tags=["files"])


@router.post("/")
def post_file(
    file: UploadFile = File(...), document_subject: str = Form(...)
) -> PipelineResult:
    if file.content_type != "application/pdf" or Path(file.filename).suffix != ".pdf":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Invalid document type, only PDF is supported",
        )
    db = Neo4jDatabase()
    llm = LLM(model_name=f"groq:{getenv('GROQ_MODEL')}")
    embedder = EmbbeddingHuggingFace()
    splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=10)
    adapter_splitter = LangChainTextSplitterAdapter(splitter)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_bytes = file.file.read()
        tmp.write(pdf_bytes)
        result = db.create_graph_from_pdf(
            llm=llm,
            embedder=embedder,
            file_path=tmp.name,
            document_metada={"subject": document_subject},
            text_splitter=adapter_splitter,
        )
    return result


@router.delete("/{document_subject}")
def delete_file_with_subject(document_subject: str) -> list[dict]:
    db = Neo4jDatabase()
    result = db.delete_document_with_metadata(metadata={"subject": document_subject})
    return result
