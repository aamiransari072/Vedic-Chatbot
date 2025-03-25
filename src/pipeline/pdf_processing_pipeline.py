import os
import uuid
from typing import List, Dict, Any, Optional
import boto3
import pymongo
from bson import ObjectId
from dotenv import load_dotenv
import pinecone
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.components.document_extraction import DocumentExtractor
from src.components.document_processing import DocumentProcessor
from src.components.data_splitter import DataSplitter
from src.components.data_embeddings import DataEmbeddings
from src.components.data_ingestion import DataIngestionService
from src.logging import logger
from langchain_core.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class PDFProcessingPipeline:
    """
    Pipeline for processing PDF documents:
    1. Upload PDFs to S3
    2. Store metadata in MongoDB
    3. Extract text from PDFs
    4. Store vectors in Pinecone
    """
    
    def __init__(self):
        """Initialize all necessary connections and components"""
        logger.info("Initializing PDFProcessingPipeline")
        
        # Initialize data ingestion service
        self.data_ingestion = DataIngestionService()
        
        # Initialize Pinecone
        pinecone.init(
            api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
        )
        self.index_name = os.environ.get("PINECONE_INDEX_NAME", "pdf-vectors")
        
        # Create index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=768,  # Default for many HuggingFace embeddings
                metric="cosine"
            )
        self.pinecone_index = pinecone.Index(self.index_name)
        logger.info(f"Pinecone initialized with index: {self.index_name}")
        
        # Initialize components
        self.document_extractor = DocumentExtractor()
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.data_splitter = DataSplitter(self.text_splitter)
        
        # Initialize embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # MongoDB collection name for PDF documents
        self.pdf_collection = "pdf_documents"
        
        # Create temp directory for document storage
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Temporary directory created at: {self.temp_dir}")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            file_path: The path to the PDF file
        
        Returns:
            text: The extracted text
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        
        text = self.document_extractor.extract_text_from_file(file_path)
        
        if text:
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
        else:
            logger.warning(f"Failed to extract text from PDF: {file_path}")
            text = ""
        
        return text
    
    def store_vectors_in_pinecone(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Store text vectors in Pinecone
        
        Args:
            text: The text to vectorize and store
            metadata: Metadata to store with the vectors
        
        Returns:
            success: Whether the operation was successful
        """
        logger.info("Storing vectors in Pinecone")
        
        try:
            # Create document
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            
            # Split into chunks
            docs = self.text_splitter.split_documents([doc])
            
            # Create vector store
            vector_store = PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings_model,
                index_name=self.index_name
            )
            
            logger.info(f"Successfully stored {len(docs)} vectors in Pinecone")
            return True
        
        except Exception as e:
            logger.error(f"Error storing vectors in Pinecone: {str(e)}")
            return False
    
    def process_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a single PDF file
        
        Args:
            file_content: The binary content of the PDF file
            filename: The name of the file
        
        Returns:
            result: Processing result with status and IDs
        """
        logger.info(f"Processing PDF: {filename}")
        
        try:
            # Upload to S3
            s3_key = self.data_ingestion.upload_file_to_s3(file_content, filename, folder="pdfs")
            
            # Save metadata to MongoDB
            metadata = {
                "filename": filename,
                "s3_key": s3_key,
                "upload_time": ObjectId().generation_time,
                "processed": False,
                "file_type": "pdf"
            }
            document_id = self.data_ingestion.save_metadata_to_mongodb(self.pdf_collection, metadata)
            
            # Save temporary file for processing
            file_path = self.data_ingestion.save_temporary_file(file_content, filename)
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(file_path)
            
            # Store vectors in Pinecone
            vector_metadata = {
                "filename": filename,
                "s3_key": s3_key,
                "document_id": document_id
            }
            vector_success = self.store_vectors_in_pinecone(text, vector_metadata)
            
            # Update MongoDB status
            self.data_ingestion.update_document_in_mongodb(
                self.pdf_collection,
                document_id,
                {"processed": vector_success}
            )
            
            # Cleanup temporary file
            self.data_ingestion.cleanup_temporary_file(file_path)
            
            return {
                "success": True,
                "document_id": document_id,
                "s3_key": s3_key,
                "vectorized": vector_success
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_multiple_pdfs(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files
        
        Args:
            files: List of dictionaries with file content and filename
                  Each dict should have 'content' and 'filename' keys
        
        Returns:
            results: List of processing results for each file
        """
        logger.info(f"Processing {len(files)} PDF files")
        
        results = []
        for file_info in files:
            result = self.process_pdf(file_info["content"], file_info["filename"])
            results.append(result)
        
        return results 