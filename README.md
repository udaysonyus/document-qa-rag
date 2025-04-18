# document-qa-rag

# 🧠 Local RAG Assistant using LlamaIndex + ChromaDB

This project demonstrates a local Retrieval-Augmented Generation (RAG) pipeline using the [LlamaIndex](https://docs.llamaindex.ai/) framework, [Ollama](https://ollama.com/) for running language models locally, and [ChromaDB](https://docs.trychroma.com/) as the vector database.

It allows you to query local documents using natural language and get intelligent answers grounded in the document content.

---

## 🚀 Features

- Loads and indexes documents from a local directory
- Embeds text using `BAAI/bge-small-en` via Hugging Face
- Uses `llama3.2` model served through Ollama
- Retrieves top 3 most relevant chunks for context-aware answers
- Applies a custom prompt to keep responses grounded in the provided documents

---


---

## ⚙️ Setup Instructions

### 1. Install Python Dependencies

bash
pip install llama-index chromadb sentence-transformers

# In your main.py, update the following:

### ChromaDB path
chroma_client = PersistentClient(path="./chroma_db")

### Vector collection name
chroma_collection = chroma_client.get_or_create_collection("my_collection")

### Document loading
documents = SimpleDirectoryReader("path_to_your_document").load_data()


#Tech Stack

LLM: llama3.2 via Ollama

Embeddings: BAAI/bge-small-en from HuggingFace

Vector Store: ChromaDB

Framework: LlamaIndex

