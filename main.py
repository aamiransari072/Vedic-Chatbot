from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import os
# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define Pinecone Index Name
index_name = 'vedic-docs'
print('index_name', index_name)




# Load the Pinecone index
vector_store = LangchainPinecone.from_existing_index(index_name, embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
))

# Query embedding
query = "What are the main challenges first-year students face when conducting academic research?"
query_vector = vector_store.embeddings.embed_query(query)

# Perform similarity search
docs = vector_store.similarity_search_with_score(query=query,k=5)

print(docs[0])

# # Print results
# for doc in docs:
#     print(doc.page_content)  # Adjust as needed


# print(pc.list_indexes().names())
