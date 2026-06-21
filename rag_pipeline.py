from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "returns_policy_pl.pdf"
ENV_PATH = BASE_DIR / ".env"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"
DEFAULT_POSTGRES_CONNECTION_STRING = (
    "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db"
)
DEFAULT_COLLECTION_NAME = "returns_policy_pl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PDF_LOADER_NAME = "PyMuPDFLoader"
LEGAL_TERM_REPLACEMENTS = {
    "Konsument": "konsument",
    "Konsumenta": "konsumenta",
    "Konsumentem": "konsumentem",
    "Konsumentowi": "konsumentowi",
    "Konsumentów": "konsumentów",
    "Produkt": "produkt",
    "Produktu": "produktu",
    "Produktem": "produktem",
    "Produkty": "produkty",
    "Produktów": "produktów",
    "Sklep Internetowy": "sklep internetowy",
    "Sklepie Internetowym": "sklepie internetowym",
    "Sprzedawca": "sprzedawca",
    "Sprzedawcy": "sprzedawcy",
    "Sprzedawcę": "sprzedawcę",
    "Sprzedawcą": "sprzedawcą",
}
LEGAL_TERM_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(term) for term in sorted(LEGAL_TERM_REPLACEMENTS, key=len, reverse=True))
    + r")\b"
)
STREAM_NORMALIZATION_BUFFER_SIZE = max(len(term) for term in LEGAL_TERM_REPLACEMENTS) + 8


def load_environment() -> None:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key":
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env before running API-dependent steps."
        )


def get_postgres_connection_string() -> str:
    load_dotenv(ENV_PATH)
    return os.getenv(
        "POSTGRES_CONNECTION_STRING",
        DEFAULT_POSTGRES_CONNECTION_STRING,
    ).strip()


def get_collection_name() -> str:
    load_dotenv(ENV_PATH)
    return os.getenv("PGVECTOR_COLLECTION_NAME", DEFAULT_COLLECTION_NAME).strip()


def get_embedding_model() -> str:
    load_dotenv(ENV_PATH)
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


def load_documents(pdf_path: Path = PDF_PATH) -> list[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def assign_chunk_ids(chunks: list[Document]) -> list[str]:
    ids = []
    for index, chunk in enumerate(chunks):
        source = Path(chunk.metadata.get("source", PDF_PATH.name)).stem
        page = chunk.metadata.get("page", 0) + 1
        chunk_id = f"{source}:page-{page}:chunk-{index}"
        chunk.id = chunk_id
        ids.append(chunk_id)
    return ids


def build_index_metadata(chunks: list[Document]) -> dict[str, object]:
    pdf_bytes = PDF_PATH.read_bytes()
    chunk_ids = [chunk.id or "" for chunk in chunks]
    config = {
        "source_file": PDF_PATH.name,
        "source_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "loader": PDF_LOADER_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "chunk_ids_sha256": hashlib.sha256(
            json.dumps(chunk_ids, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "embedding_model": get_embedding_model(),
    }
    config["index_signature"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return config


def create_embeddings() -> OpenAIEmbeddings:
    load_environment()
    return OpenAIEmbeddings(model=get_embedding_model())


def get_indexed_collection_metadata(
    connection_string: str,
    collection_name: str,
) -> tuple[int, dict[str, object] | None]:
    engine = create_engine(connection_string)
    query = text(
        """
        SELECT
            (
                SELECT COUNT(e.id)
                FROM langchain_pg_embedding e
                WHERE e.collection_id = c.uuid
            ) AS indexed_count,
            c.cmetadata
        FROM langchain_pg_collection c
        WHERE c.name = :collection_name
        """
    )
    with engine.connect() as connection:
        result = connection.execute(query, {"collection_name": collection_name})
        row = result.first()
        if not row:
            return 0, None
        return int(row[0]), row[1]


def connect_vector_store(embeddings: OpenAIEmbeddings) -> PGVector:
    return PGVector(
        embeddings=embeddings,
        connection=get_postgres_connection_string(),
        collection_name=get_collection_name(),
        use_jsonb=True,
    )


def get_or_create_vector_store(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
    recreate: bool = False,
) -> PGVector:
    connection_string = get_postgres_connection_string()
    collection_name = get_collection_name()
    assign_chunk_ids(chunks)
    index_metadata = build_index_metadata(chunks)

    vector_store = PGVector(
        embeddings=embeddings,
        connection=connection_string,
        collection_name=collection_name,
        collection_metadata=index_metadata,
        pre_delete_collection=False,
        use_jsonb=True,
    )

    indexed_count, existing_metadata = get_indexed_collection_metadata(
        connection_string,
        collection_name,
    )
    existing_signature = (existing_metadata or {}).get("index_signature")
    target_signature = index_metadata["index_signature"]
    should_recreate = recreate or existing_signature != target_signature

    if should_recreate:
        if indexed_count:
            print("Detected a change in the PDF, chunking settings, loader, or embedding model.")
        print(f"Rebuilding pgvector collection: {collection_name}")
        vector_store.delete_collection()
        vector_store = PGVector(
            embeddings=embeddings,
            connection=connection_string,
            collection_name=collection_name,
            collection_metadata=index_metadata,
            pre_delete_collection=False,
            use_jsonb=True,
        )
        indexed_count = 0

    if indexed_count:
        print(f"Loading existing pgvector collection: {collection_name} ({indexed_count} records)")
        return vector_store

    print(f"Indexing documents in PostgreSQL/pgvector: {collection_name}")
    vector_store.add_documents(chunks, ids=[chunk.id for chunk in chunks])
    print(f"Indexed {len(chunks)} chunks")
    return vector_store


def build_retriever(vector_store: PGVector, k: int = 3):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def format_docs(documents: Iterable[Document]) -> str:
    return "\n\n".join(
        f"[page {doc.metadata.get('page', 0) + 1}] {doc.page_content}"
        for doc in documents
    )


def _starts_sentence(text: str, position: int) -> bool:
    prefix = text[:position].rstrip()
    return not prefix or prefix[-1] in ".!?\n"


def normalize_legal_casing(text: str, prefix: str = "") -> str:
    full_text = prefix + text
    prefix_length = len(prefix)

    def replace(match: re.Match[str]) -> str:
        replacement = LEGAL_TERM_REPLACEMENTS[match.group(0)]
        if _starts_sentence(full_text, match.start()):
            return replacement[:1].upper() + replacement[1:]
        return replacement

    normalized = LEGAL_TERM_PATTERN.sub(replace, full_text)
    return normalized[prefix_length:]


def stream_normalized_text(chunks: Iterable[str]) -> Iterable[str]:
    buffer = ""
    history = ""

    for chunk in chunks:
        buffer += chunk
        if len(buffer) <= STREAM_NORMALIZATION_BUFFER_SIZE:
            continue

        emit_limit = len(buffer) - STREAM_NORMALIZATION_BUFFER_SIZE
        emit_length = buffer.rfind(" ", 0, emit_limit)
        if emit_length <= 0:
            continue
        emit_length += 1
        text_to_emit = buffer[:emit_length]
        normalized = normalize_legal_casing(text_to_emit, prefix=history)
        yield normalized
        history = (history + normalized)[-120:]
        buffer = buffer[emit_length:]

    if buffer:
        yield normalize_legal_casing(buffer, prefix=history)


def build_rag_chain(retriever):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    prompt_template = """You are an assistant that answers questions about a return policy.
Answer using only the context below.
If the context does not contain the answer, respond exactly:
"Nie znam odpowiedzi na to pytanie na podstawie dostepnej polityki zwrotow."
Use natural sentence casing in the answer. Do not preserve legal capitalization from the source unless it is part of a proper name.
Always cite the page number in square brackets, for example: [page 2].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_ready_rag_chain():
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = get_or_create_vector_store(chunks, embeddings)
    retriever = build_retriever(vector_store)
    return build_rag_chain(retriever)
