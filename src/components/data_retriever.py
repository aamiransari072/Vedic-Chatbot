from typing import List, Dict, Any, Optional, Tuple
import os
import pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.logging import logger

load_dotenv()

class DataRetriever:
    """
    Component for retrieving relevant documents from vector databases
    based on user queries.
    """
    
    def __init__(self, index_name: Optional[str] = None, top_k: int = 5):
        """
        Initialize the data retriever with vector database connections
        
        Args:
            index_name: Name of the Pinecone index (default: from env or "pdf-vectors")
            top_k: Number of documents to retrieve per query
        """
        logger.info("Initializing DataRetriever")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
        )
        
        # Set index name
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "pdf-vectors")
        logger.info(f"Using Pinecone index: {self.index_name}")
        
        # Set retrieval parameters
        self.top_k = top_k
        logger.info(f"Will retrieve top {top_k} documents per query")
        
        # Initialize embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        logger.info("Embeddings model initialized")
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings_model
        )
        logger.info("Vector store initialized")
    
    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve documents relevant to the query
        
        Args:
            query: The user's query string
            top_k: Override for number of documents to retrieve
                (default: use instance value)
        
        Returns:
            List of retrieved documents
        """
        k = top_k or self.top_k
        logger.info(f"Retrieving top {k} documents for query: {query}")
        
        try:
            # Get documents from vector store
            docs = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: The user's query string
            top_k: Override for number of documents to retrieve
                (default: use instance value)
        
        Returns:
            List of tuples with (document, score)
        """
        k = top_k or self.top_k
        logger.info(f"Retrieving top {k} documents with scores for query: {query}")
        
        try:
            # Get documents from vector store with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"Retrieved {len(docs_and_scores)} documents with scores")
            return docs_and_scores
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def retrieve_and_rerank(self, query: str, top_k_retrieve: int = 10, top_k_rerank: int = 5) -> List[Document]:
        """
        Retrieve more documents and rerank them based on relevance to query
        (Simple implementation that just sorts by score)
        
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
            docs_and_scores = self.retrieve_with_scores(query, top_k=top_k_retrieve)
            
            # Sort by score (higher is better)
            sorted_docs = sorted(docs_and_scores, key=lambda x: x[1], reverse=True)
            
            # Get just the documents
            reranked_docs = [doc for doc, _ in sorted_docs[:top_k_rerank]]
            
            logger.info(f"Retrieved and reranked to {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error retrieving and reranking documents: {str(e)}")
            return [] 