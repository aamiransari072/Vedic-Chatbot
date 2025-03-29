from typing import List, Dict, Any, Optional, Tuple
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.logging import logger
from src.utils.env_checker import check_required_env_vars

load_dotenv()

class DataRetriever:
    """
    Component for retrieving relevant documents from Pinecone vector database
    based on user queries.
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize DataRetriever with vector store connection
        
        Args:
            top_k: Default number of documents to retrieve
        """
        # Check required environment variables
        check_required_env_vars()
        
        # Set default number of documents to retrieve
        self.top_k = top_k
        
        # Initialize Pinecone client
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
        )
        
        # Get Pinecone index
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable is required")
            
        # Get the Pinecone index
        index = self.pc.Index(index_name)
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text"
        )
        
        logger.info(f"DataRetriever initialized with index '{index_name}'")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to the query using similarity search
        
        Args:
            query: The user's query string
        
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        try:
            # Get documents from vector store
            docs = self.vector_store.similarity_search(
                query=query, 
                k=self.top_k
            )
            
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: The user's query string
        
        Returns:
            List of tuples with (document, score)
        """
        logger.info(f"Retrieving documents with scores for query: {query}")
        
        try:
            # Get documents from vector store with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query=query, 
                k=self.top_k
            )
            
            logger.info(f"Retrieved {len(docs_and_scores)} documents with scores")
            return docs_and_scores
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def retrieve_and_rerank(self, query: str, top_k_retrieve: int = 10, 
                           top_k_rerank: int = 5) -> List[Document]:
        """
        Retrieve more documents and rerank them based on relevance to query
        
        Args:
            query: The user's query string
            top_k_retrieve: Number of documents to initially retrieve
            top_k_rerank: Number of documents to return after reranking
        
        Returns:
            List of reranked documents
        """
        logger.info(f"Retrieving and reranking documents for query: {query}")
        
        try:
            # Get documents with scores
            docs_and_scores = self.retrieve_with_scores(query=query)
            
            # Sort by score (higher is better)
            sorted_docs = sorted(docs_and_scores, key=lambda x: x[1], reverse=True)
            
            # Get just the documents
            reranked_docs = [doc for doc, _ in sorted_docs[:top_k_rerank]]
            
            logger.info(f"Retrieved and reranked to {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error retrieving and reranking documents: {str(e)}")
            return [] 