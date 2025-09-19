import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings # Or any other local embedding model

# Load documents from the Resources directory
loader = DirectoryLoader('Resources', glob="**/*.pdf") # Update glob for other file types
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create a Chroma vector store
# Replace with your preferred embedding model
embeddings = OllamaEmbeddings()
db = Chroma.from_documents(texts, embeddings)

print("Vector store created successfully!")

# Example of how to query the vector store
query = "What is the main topic of the documents?"
retriever = db.as_retriever()
docs = retriever.get_relevant_documents(query)

print(f"Query: {query}")
for doc in docs:
    print(doc.page_content)
