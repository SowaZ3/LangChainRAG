from rag_pipeline import load_documents, split_documents


documents = load_documents()
chunks = split_documents(documents)

average_length = sum(len(chunk.page_content) for chunk in chunks) // len(chunks)

print(f"Split into {len(chunks)} chunks")
print(f"Average length: {average_length} characters")
