from rag_pipeline import (
    build_retriever,
    create_embeddings,
    get_or_create_vector_store,
    load_documents,
    split_documents,
)


documents = load_documents()
chunks = split_documents(documents)
embeddings = create_embeddings()
vector_store = get_or_create_vector_store(chunks, embeddings)
retriever = build_retriever(vector_store)

test_docs = retriever.invoke("Jaki jest termin zwrotu produktu?")
for index, document in enumerate(test_docs, start=1):
    page_number = document.metadata.get("page", 0) + 1
    print(f"\n=== Result {index} (page {page_number}) ===")
    print(document.page_content[:300])
