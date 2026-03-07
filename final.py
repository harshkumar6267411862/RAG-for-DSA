from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
import json
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# -----------------------------
# Vector DB connection
# -----------------------------

persistent_directory = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# -----------------------------
# Local LLM (CPU friendly)
# -----------------------------

pipe = pipeline(
    task="text-generation",
    model="google/flan-t5-small",
    max_new_tokens=200
)

model = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Chat memory
# -----------------------------

chat_history = []

# -----------------------------
# Answer validation
# -----------------------------

def validate_and_format_answer(answer):

    validation_report = []

    if "```" in answer:
        validation_report.append("Code block detected.")
    else:
        validation_report.append("No formatted code block found.")

    complexity_pattern = r"O\([^)]+\)"
    complexities = re.findall(complexity_pattern, answer)

    if complexities:
        validation_report.append(f"Time complexity mentioned: {', '.join(complexities)}")
    else:
        validation_report.append("No time complexity mentioned.")

    formatted_answer = answer.strip()

    return formatted_answer, validation_report


# -----------------------------
# Logging interactions
# -----------------------------

def log_interaction(question, search_question, docs, answer, validation):

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_question": question,
        "search_query": search_question,
        "retrieved_sources": [
            doc.metadata.get("source", "Unknown") for doc in docs
        ],
        "answer": answer,
        "validation": validation
    }

    try:
        with open("rag_logs.json", "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(log_entry)

    with open("rag_logs.json", "w") as f:
        json.dump(data, f, indent=4)


# -----------------------------
# Analytics
# -----------------------------

def show_analytics():

    try:
        with open("rag_logs.json", "r") as f:
            data = json.load(f)
    except:
        print("No logs available yet.")
        return

    print("\n--- Analytics ---")
    print("Total Queries:", len(data))

    sources = {}

    for entry in data:
        for src in entry["retrieved_sources"]:
            sources[src] = sources.get(src, 0) + 1

    print("\nMost Retrieved Sources:")

    for k, v in sources.items():
        print(k, ":", v)


# -----------------------------
# Main RAG logic
# -----------------------------

def ask_question(user_question):

    print(f"\n--- You asked: {user_question} ---")

    # Rewrite question using chat history
    if chat_history:

        history_text = "\n".join(
            [msg.content for msg in chat_history]
        )

        rewrite_prompt = f"""
Rewrite the user question so it can be used for document search.

Chat history:
{history_text}

User question:
{user_question}

Rewritten query:
"""

        search_question = model.invoke(rewrite_prompt).strip()

        print(f"Searching for: {search_question}")

    else:
        search_question = user_question

    # Retrieve documents
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents")

    docs_text = "\n\n".join([doc.page_content[:800] for doc in docs])

    # Prompt for answer generation
    prompt = f"""
You are a helpful Data Structures tutor.

Use the retrieved documents to answer the question clearly.

If the documents contain relevant information, use them.
If they do not contain the answer, you may use your general knowledge.

Question:
{user_question}

Documents:
{docs_text}

Give a clear and concise explanation:
"""

    answer = model.invoke(prompt).strip()

    answer, validation = validate_and_format_answer(answer)

    print("\nValidation Report:")
    for item in validation:
        print("-", item)

    log_interaction(user_question, search_question, docs, answer, validation)

    # Update chat history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    chat_history[:] = chat_history[-6:]

    print("\nAnswer:\n", answer)

    return answer


# -----------------------------
# CLI chat (optional)
# -----------------------------

def start_chat():

    print("Ask questions. Type 'quit' to exit. Type 'analytics' for stats.")

    while True:

        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if question.lower() == "analytics":
            show_analytics()
            continue

        ask_question(question)


if __name__ == "__main__":
    start_chat()
