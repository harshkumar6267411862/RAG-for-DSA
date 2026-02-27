from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import pipeline

persistent_directory = "db/chroma_db"

# ---------------------------
# Embedding Model (LOCAL)
# ---------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ---------------------------
# Load Vector Store
# ---------------------------
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ---------------------------
# Retriever
# ---------------------------
query = "Explain the difference between singly and doubly linked lists?"

retriever = db.as_retriever(search_kwargs={"k": 2})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("\n--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:\n{doc.page_content}\n")

# ---------------------------
# Combine Context
# ---------------------------
combined_input = f"""
Answer the question using ONLY the provided documents.

Question:
{query}

Documents:
{chr(10).join([doc.page_content for doc in relevant_docs])}

If the answer is not contained in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""

# ---------------------------
# Local HuggingFace LLM
# ---------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)

model = HuggingFacePipeline(pipeline=pipe)

# ---------------------------
# Generate Response
# ---------------------------
result = model.invoke(combined_input)

print("\n--- Generated Response ---")
print(result)