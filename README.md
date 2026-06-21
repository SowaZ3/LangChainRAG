<p align="center">
  <img src="./screens/header.png" alt="photo" width="50%">
</p>

<h1 align="center">LangChain RAG for returns_policy_pl.pdf with PostgreSQL/pgvector</h1>

This project is a local Retrieval-Augmented Generation application for answering questions about the `returns_policy_pl.pdf` return policy. It reads configuration from `.env`, stores document embeddings in PostgreSQL with the `pgvector` extension, retrieves relevant policy fragments, and streams answers from an OpenAI chat model.

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start PostgreSQL with pgvector:

```powershell
docker compose up -d
```

If Docker reports a Docker API connection error, start Docker Desktop and run the command again.

4. Create `.env` from the example file:

```powershell
Copy-Item .env.example .env
```

5. Fill in `.env`:

```env
OPENAI_API_KEY=your_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
POSTGRES_CONNECTION_STRING=postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db
PGVECTOR_COLLECTION_NAME=returns_policy_pl
```

## Pipeline Steps

- `python 01_load_pdf.py`
- `python 02_split_chunks.py`
- `python 03_create_embeddings.py`
- `python 04_index_pgvector.py`
- `python 05_test_retriever.py`
- `python 06_build_chain.py`
- `python 07_ask_questions.py`

The `04_index_pgvector.py` step checks whether the index needs to be rebuilt. Reindexing runs automatically when the PDF, PDF loader, chunking settings, or embedding model changes.

Quick end-to-end test after adding the API key:

```powershell
docker compose up -d
python 04_index_pgvector.py
python 07_ask_questions.py
```

The `07_ask_questions.py` step first asks predefined default questions and streams the answers. After that, it switches to interactive mode where you can type questions from the keyboard.

## Project Notes

- Shared logic lives in `rag_pipeline.py`.
- The project reads the API key from `.env`.
- Embeddings use `text-embedding-3-large` by default, which is a better fit for Polish documents.
- The vector store uses PostgreSQL/pgvector.
- PDF text is read with PyMuPDF to reduce incorrect spaces inside Polish words.
- Answers are based only on `returns_policy_pl.pdf` and cite page numbers.
- Streaming output is handled in `07_ask_questions.py`.

## Problems and Fixes

### Legal capitalization copied from the source

The source PDF uses legal-style capitalization for defined terms such as `Produkt`, `Sprzedawca`, `Konsument`, and `Sklep Internetowy`. The model often copied that style into normal answers, producing capitalized words in the middle of a sentence.

Example before the fix:

![Answer with legal capitalization highlighted](./screens/1.png)

The prompt now asks the model to use natural sentence casing:

```text
Use natural sentence casing in the answer. Do not preserve legal capitalization from the source unless it is part of a proper name.
```

Because prompt instructions are not always enough, `rag_pipeline.py` also applies a small post-processing step with `normalize_legal_casing()` and `stream_normalized_text()`. This keeps streaming output readable while converting legal terms such as `Produktu` to `produktu` when they appear in the middle of a sentence.

Example after the fix:

![Answer after legal capitalization normalization](./screens/2.png)

### Typos in user questions

The retriever and model can still answer correctly when the question contains a small typo, as long as the intent is clear and the relevant information is present in the indexed return policy.

Example:

![Question with a typo and a correct answer](./screens/3.png)

### PDF text extraction artifacts

The first version used a PDF loader that sometimes inserted spaces inside Polish words. The project now uses `PyMuPDFLoader`, which extracts cleaner text from `returns_policy_pl.pdf`.

### Docker is not running

If `docker compose up -d` fails with a Docker API connection error, Docker Desktop is usually not running yet. Start Docker Desktop, wait until Docker is ready, and run the command again.
