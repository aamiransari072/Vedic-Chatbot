import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.components.data_retriever import DataRetriever
from src.Agent.google import Gemini
from src.logging import logger
from src.utils.environment import get_env_variable


load_dotenv()

# Default prompt template for answering queries
DEFAULT_PROMPT_TEMPLATE = """
You are a helpful and knowledgeable assistant who specializes in Vedic knowledge.

Context information is provided below. Given the context information and not prior knowledge, 
answer the user's question to the best of your ability.

If the answer is not provided in the context information, politely state that you don't have enough information,
but try to provide related information that might be helpful.

Context information:
{context}

User Question: {question}

Your Answer:
"""

class QueryPipeline:
    """
    Pipeline for answering user queries using document retrieval and LLM:
    1. Retrieve relevant documents based on user query
    2. Construct context from retrieved documents
    3. Send query and context to LLM for answering
    """
    
    def __init__(self, top_k: int = 5, prompt_template: str = None):
        """
        Initialize the query pipeline with retriever and LLM
        
        Args:
            top_k: Number of documents to retrieve per query
            prompt_template: Custom prompt template (if None, use default)
        """
        logger.info("Initializing QueryPipeline")
        
        # Initialize the data retriever
        self.retriever = DataRetriever(top_k=top_k)
        logger.info(f"Data retriever initialized with top_k={top_k}")
        
        # Initialize the Gemini LLM
        google_api_key = get_env_variable("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.llm = Gemini(api_key=google_api_key)
        logger.info("Gemini LLM initialized")
        
        # Set up the prompt template
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        logger.info("Prompt template initialized")
    
    def format_documents(self, docs: List[Document]) -> str:
        """
        Format a list of documents into a single context string
        
        Args:
            docs: List of retrieved documents
        
        Returns:
            Formatted context string
        """
        if not docs:
            return "No relevant documents found."
        
        # Format each document with its content
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_doc = f"[Document {i+1}]\n\nContent:\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        
        # Join all formatted documents
        return "\n".join(formatted_docs)
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a user query using retrieval and LLM
        
        Args:
            query: The user's query string
        
        Returns:
            Dictionary with query results including answer and sources
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.retrieve_documents(query=query)
            if not docs:
                logger.warning("No documents retrieved for query")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "success": False
                }
            
            # Format documents into context
            context = self.format_documents(docs)
            logger.info(f"Created context from {len(docs)} documents")
            
            # Prepare the prompt with context and query
            full_prompt = self.prompt.format(context=context, question=query)
            
            # Generate response using Gemini
            response = self.llm.generate(full_prompt)
            answer = response.text
            logger.info("Generated answer using Gemini LLM")
            
            # Prepare source information
            sources = []
            for doc in docs:
                source_info = {
                    "document_id": doc.metadata.get("document_id", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown")
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def answer_query_with_reranking(self, query: str, top_k_retrieve: int = 10, 
                                    top_k_rerank: int = 5) -> Dict[str, Any]:
        """
        Answer a user query with reranking of retrieved documents
        
        Args:
            query: The user's query string
            top_k_retrieve: Number of documents to initially retrieve
            top_k_rerank: Number of documents to use after reranking
        
        Returns:
            Dictionary with query results including answer and sources
        """
        logger.info(f"Processing query with reranking: {query}")
        
        try:
            # Retrieve and rerank documents
            docs = self.retriever.retrieve_and_rerank(
                query=query, 
                top_k_retrieve=top_k_retrieve, 
                top_k_rerank=top_k_rerank
            )
            
            if not docs:
                logger.warning("No documents retrieved for query")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "success": False
                }
            
            # Format documents into context
            context = self.format_documents(docs)
            logger.info(f"Created context from {len(docs)} reranked documents")
            
            # Prepare the prompt with context and query
            full_prompt = self.prompt.format(context=context, question=query)
            
            # Generate response using Gemini
            response = self.llm.generate(full_prompt)
            answer = response.text
            logger.info("Generated answer using Gemini LLM")
            
            # Prepare source information
            sources = []
            for doc in docs:
                source_info = {
                    "document_id": doc.metadata.get("document_id", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown")
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error answering query with reranking: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "success": False
            } 