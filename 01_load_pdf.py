from rag_pipeline import load_documents


documents = load_documents()

print(f"Loaded {len(documents)} pages")
print(f"First page ({len(documents[0].page_content)} characters):")
print(documents[0].page_content[:300])
