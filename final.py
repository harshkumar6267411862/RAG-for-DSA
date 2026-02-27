from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
import json
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up lightweight AI model (CPU friendly)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)

model = HuggingFacePipeline(pipeline=pipe)

# Store our conversation as messages
chat_history = []

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

def log_interaction(question, search_question, docs, answer, validation):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_question": question,
        "search_query": search_question,
        "retrieved_sources": [doc.metadata.get("source", "Unknown") for doc in docs],
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

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    combined_input = f"""Answer the question using ONLY the provided documents.

Question:
{user_question}

Documents:
{"\n".join([doc.page_content for doc in docs])}

If the answer is not in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""
    
    answer = model.invoke(combined_input)
    answer, validation = validate_and_format_answer(answer)

    print("\nValidation Report:")
    for item in validation:
        print("-", item)

    log_interaction(user_question, search_question, docs, answer, validation)

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    chat_history[:] = chat_history[-4:]
    
    print(f"\nAnswer:\n{answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit. Type 'analytics' to see usage stats.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        
        if question.lower() == 'analytics':
            show_analytics()
            continue
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()