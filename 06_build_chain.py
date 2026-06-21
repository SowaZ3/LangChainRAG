from rag_pipeline import (
    build_rag_chain,
    build_retriever,
    create_embeddings,
    get_or_create_vector_store,
    load_documents,
    normalize_legal_casing,
    split_documents,
)


documents = load_documents()
chunks = split_documents(documents)
embeddings = create_embeddings()
vector_store = get_or_create_vector_store(chunks, embeddings)
retriever = build_retriever(vector_store)
rag_chain = build_rag_chain(retriever)

answer = rag_chain.invoke("Czy moge zwrocic produkt kupiony online?")
print(normalize_legal_casing(answer))
