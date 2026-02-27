# Conversational RAG Question Answering System

A Retrieval-Augmented Generation (RAG) based conversational question answering system built using:

- LangChain
- ChromaDB (Vector Database)
- Hugging Face Models
- Sentence Transformers (Embeddings)

This project demonstrates how to build a grounded AI system that answers questions strictly based on provided documents.

---

## 🚀 What is RAG?

RAG (Retrieval-Augmented Generation) combines:

1. **Retrieval** – Searching relevant documents from a knowledge base  
2. **Generation** – Using a language model to generate answers based on retrieved documents  

Instead of relying only on a model’s training data, this system retrieves relevant information first and then generates a response grounded in those documents.

This significantly reduces hallucinations and improves reliability.

---

## 🧠 How This Project Works

### 1️⃣ Document Storage
Documents are stored inside a persistent ChromaDB directory.  
Each document is converted into vector embeddings using a Hugging Face embedding model.

---

### 2️⃣ Conversational Context Handling
If the user asks follow-up questions:
- The system uses chat history
- Rewrites the new question into a standalone version
- Makes it searchable

This ensures better retrieval accuracy.

---

### 3️⃣ Document Retrieval
The system retrieves the top 3 most relevant documents using vector similarity search.

Only those documents are passed to the language model.

---

### 4️⃣ Controlled Answer Generation
The model is instructed to:
- Answer only from provided documents
- Clearly state if the answer cannot be found

This keeps responses grounded and trustworthy.

---

## 🛠 Tech Stack

- **LangChain** – Orchestration framework
- **ChromaDB** – Vector database for document storage
- **Hugging Face Transformers** – Language model
- **Sentence Transformers** – Embeddings model
- **Python** – Core implementation

---

## 📂 Project Structure

```
.
├── db/
│   └── chroma_db/        # Persistent vector database
├── main.py               # RAG pipeline implementation
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python main.py
```

You will see:

```
Ask me questions! Type 'quit' to exit.
```

Start asking questions related to your stored documents.

---

## ❗ Challenges Faced

- Token length limitations
- Managing conversation history efficiently
- Switching from OpenAI API to local Hugging Face models
- Preventing hallucinations
- Controlling generation parameters

These were resolved through:
- Proper chunking strategy
- Limiting retrieved documents
- Prompt control
- Pipeline configuration adjustments

---

## 🔮 Future Improvements

- Semantic chunking
- Hybrid search (keyword + vector)
- Document re-ranking
- Source citations in answers
- Web UI interface
- Cloud deployment

---

## 📌 Key Learnings

- How vector databases work
- Importance of embedding quality
- Prompt engineering for RAG
- Managing conversational memory
- Designing reliable AI pipelines

---

## 🏁 Conclusion

This project demonstrates how to build a grounded conversational AI system using Retrieval-Augmented Generation.

It moves beyond simple LLM usage and focuses on building reliable, scalable, and domain-aware AI systems.

---

## 📜 License

This project is open-source and available under the MIT License.
