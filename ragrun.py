import chromadb
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient  
# -------------------------------
# Step 1: Configure LLM and Embedding
# -------------------------------
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
# -------------------------------
# Step 2: Initialize ChromaDB
# -------------------------------
chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("Your Collection Name")
# -------------------------------
# Step 3: Load Documents
# -------------------------------

documents = SimpleDirectoryReader("./data1/").load_data()
# -------------------------------
# Step 4: Set up Vector Store and Storage
# -------------------------------
vector_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# -------------------------------
# Step 5: Build the Index
# -------------------------------
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    embed_model=embed_model)
# -------------------------------
# Step 6: Define Custom Prompt (general-purpose)
# -------------------------------
custom_prompt = PromptTemplate(
    template="""
You are a helpful and intelligent assistant. Use only the context provided below to answer the user's question.

If the answer is not found in the context, say:
"The information is not available in the provided context."

--- CONTEXT ---
{context_str}

--- QUESTION ---
{query_str}

--- ANSWER ---
"""
)
# -------------------------------
# Step 7: Create Query Engine
# -------------------------------
query_engine = index.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=3, # Retrieves the top 3 most relevant chunks from the vector store to use as context for answering the query.
    text_qa_template=custom_prompt)
# -------------------------------
# Step 8: Take user input and answer
# -------------------------------
user_query = input("\nPlease type your question : ")
response = query_engine.query(user_query)

print("\nAnswer:", response)
