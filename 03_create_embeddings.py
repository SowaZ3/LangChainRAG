from rag_pipeline import create_embeddings, load_documents, split_documents


documents = load_documents()
chunks = split_documents(documents)
embeddings = create_embeddings()

sample_vector = embeddings.embed_query(chunks[0].page_content)
print(f"Vector has {len(sample_vector)} dimensions")
