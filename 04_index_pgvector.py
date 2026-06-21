from rag_pipeline import create_embeddings, get_or_create_vector_store, load_documents, split_documents


documents = load_documents()
chunks = split_documents(documents)
embeddings = create_embeddings()
vector_store = get_or_create_vector_store(chunks, embeddings)

print(f"Vector store ready: {vector_store.__class__.__name__}")
