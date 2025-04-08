import os
import uuid
import io
import magic
from typing import List, Dict, Any, Optional
import boto3
import pymongo
from bson import ObjectId
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
import itertools

from src.components.document_extraction import DocumentExtractor
from src.components.data_splitter import DataSplitter
from src.components.data_ingestion import DataIngestionService
from src.logging import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.env_checker import check_required_env_vars
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

class PDFProcessingPipeline:
    """
    Pipeline for processing PDF documents:
    1. Store PDFs in temp directory
    2. Upload PDFs to S3
    3. Store metadata in MongoDB
    4. Load documents using PyPDFDirectoryLoader
    5. Split documents into chunks
    6. Create embeddings
    7. Store vectors in Pinecone
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize all necessary connections and components
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        logger.info("Initializing PDFProcessingPipeline")
        
        # Check required environment variables
        check_required_env_vars()
        
        # Initialize data ingestion service
        self.data_ingestion = DataIngestionService()
        
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
            pool_threads=30  # Enable parallel processing
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "vedic-docs")
        
        # Get embedding model dimensions
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
        )
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            # Create a sample embedding to determine dimensions
            sample_embedding = self.embeddings.embed_query("Sample text for dimension check")
            embedding_dim = len(sample_embedding)
            logger.info(f"Embedding model produces vectors with {embedding_dim} dimensions")
            
            self.pc.create_index(
                self.index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            

        # Connect to Pinecone Index
        self.pinecone_index = self.pc.Index(self.index_name)
        logger.info(f"Pinecone initialized with index: {self.index_name}")
        
        # Initialize document extractor
        # self.document_extractor = DocumentExtractor()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.data_splitter = DataSplitter(self.text_splitter)
        
        # MongoDB collection name for PDF documents
        self.pdf_collection = "documents"
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Create temp directory for document storage
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Temporary directory created at: {self.temp_dir}")
        
        # Initialize file type checker
        self.mime = magic.Magic(mime=True)

    def validate_pdf(self, file_content: bytes) -> bool:
        """
        Validate if the file content is a valid PDF
        
        Args:
            file_content: The binary content of the file
            
        Returns:
            bool: True if valid PDF, False otherwise
        """
        try:
            # Check MIME type
            mime_type = self.mime.from_buffer(file_content)
            if mime_type != 'application/pdf':
                logger.warning(f"Invalid file type: {mime_type}")
                return False
                
            # Try to read the PDF
            pdf_stream = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_stream)
            
            # Check if PDF has pages
            if len(pdf_reader.pages) == 0:
                logger.warning("PDF has no pages")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF: {str(e)}")
            return False

    def cleanup_failed_processing(self, document_id: str, s3_key: str, temp_file_path: str = None) -> None:
        """
        Clean up resources if PDF processing fails
        
        Args:
            document_id: The MongoDB document ID
            s3_key: The S3 key of the uploaded file
            temp_file_path: The path to the temporary file, if any
        """
        try:
            # Delete from S3
            self.data_ingestion.s3_client.delete_object(
                Bucket=self.data_ingestion.bucket_name,
                Key=s3_key
            )
            
            # Delete from MongoDB
            self.data_ingestion.db[self.pdf_collection].delete_one(
                {"_id": ObjectId(document_id)}
            )
            
            # Delete temp file if exists
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            logger.info(f"Cleaned up failed processing for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up failed processing: {str(e)}")

    def save_pdf_to_temp(self, file_content: bytes, filename: str) -> str:
        """
        Save PDF content to a temporary file
        
        Args:
            file_content: The binary content of the PDF file
            filename: The name of the file
            
        Returns:
            temp_file_path: Path to the saved temporary file
        """
        # Generate a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{filename}"
        temp_file_path = os.path.join(self.temp_dir, unique_filename)
        
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
            
        logger.info(f"Saved PDF to temporary file: {temp_file_path}")
        return temp_file_path

    def load_documents_from_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load documents from a PDF file using PyPDFDirectoryLoader
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            documents: List of Document objects
        """
        try:
            # Create a temporary directory for this specific PDF
            pdf_dir = os.path.dirname(pdf_path)
            
            # Use PyPDFDirectoryLoader to load documents
            loader = PyPDFDirectoryLoader(pdf_dir, glob="**/" + os.path.basename(pdf_path))
            documents = loader.load()
            
            logger.info(f"Loaded {len(documents)} documents from PDF: {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from PDF: {str(e)}")
            return []

    def chunks(self, iterable, batch_size=200):
        """Helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

    def process_documents(self, documents: List[Document], document_id: str, filename: str, s3_key: str) -> bool:
        """
        Process documents by splitting and storing in Pinecone using parallel batch upserts
        
        Args:
            documents: List of Document objects
            document_id: MongoDB document ID
            filename: Original filename
            s3_key: S3 key for the uploaded file
            
        Returns:
            success: Whether processing was successful
        """
        try:
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "filename": filename,
                    "s3_key": s3_key,
                    "document_id": document_id,
                })
            
            # Split documents into chunks
            chunks = self.data_splitter.split_data(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            
            # Prepare vectors for batch processing
            vectors_to_upsert = []

            logger.info(f"Preparing to upsert {len(chunks)} chunks for document ID: {document_id}")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_{i}"

                # Generate embeddings
                embedding_vector = self.embeddings.embed_query(chunk.page_content)
                logger.debug(f"Generated embedding for chunk {i}, ID: {chunk_id}")

                # Prepare metadata
                metadata = {
                    "filename": filename,
                    "s3_key": s3_key,
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk.page_content,
                    "page_number": chunk.metadata.get("page", 0),
                    "source": chunk.metadata.get("source", "")
                }

                # Clean metadata
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool, list, dict)) and key != "values":
                        clean_metadata[key] = value

                # Add to vectors list
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding_vector,
                    "metadata": clean_metadata
                })

            logger.info(f"Prepared {len(vectors_to_upsert)} vectors for upserting to index: {self.index_name}")

            # Process vectors in parallel batches
            with self.pc.Index(self.index_name, pool_threads=30) as index:
                async_results = []
                for i, vectors_chunk in enumerate(self.chunks(vectors_to_upsert, batch_size=200)):
                    logger.info(f"Upserting batch {i+1} with {len(vectors_chunk)} vectors...")
                    result = index.upsert(vectors=vectors_chunk, async_req=True)
                    async_results.append(result)

                try:
                    [async_result.get() for async_result in async_results]
                    logger.info(f"Successfully upserted all vectors for document {document_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error in parallel upsert for document {document_id}: {str(e)}")
                    return False
            
        except Exception as e:
            logger.error(f"Error processing document chunks: {str(e)}")
            return False

    def process_pdf_stream(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a single PDF file directly from memory
        
        Args:
            file_content: The binary content of the PDF file
            filename: The name of the file
        
        Returns:
            result: Processing result with status and IDs
        """
        logger.info(f"Processing PDF stream: {filename}")
        temp_file_path = None
        
        try:
            # Validate PDF
            if not self.validate_pdf(file_content):
                return {
                    "success": False,
                    "error": "Invalid PDF file"
                }
            
            # Save PDF to temp directory
            temp_file_path = self.save_pdf_to_temp(file_content, filename)
            
            # Upload to S3 directly
            s3_key = self.data_ingestion.upload_file_to_s3(file_content, filename, folder="pdfs")
            
            # Save metadata to MongoDB
            metadata = {
                "filename": filename,
                "s3_key": s3_key,
                "upload_time": ObjectId().generation_time,
                "processed": False,
                "file_type": "pdf",
                "temp_path": temp_file_path
            }
            document_id = self.data_ingestion.save_metadata_to_mongodb(self.pdf_collection, metadata)
            
            # Load documents from PDF
            documents = self.load_documents_from_pdf(temp_file_path)
            
            if not documents:
                logger.warning(f"No documents loaded from PDF {filename}")
                self.cleanup_failed_processing(document_id, s3_key, temp_file_path)
                return {
                    "success": False,
                    "document_id": document_id,
                    "s3_key": s3_key,
                    "vectorized": False,
                    "error": "Failed to load documents from PDF"
                }
            
            # Process documents
            processing_success = self.process_documents(documents, document_id, filename, s3_key)
            
            if not processing_success:
                self.cleanup_failed_processing(document_id, s3_key, temp_file_path)
                return {
                    "success": False,
                    "document_id": document_id,
                    "s3_key": s3_key,
                    "vectorized": False,
                    "error": "Failed to process documents"
                }
            
            # Update MongoDB status
            self.data_ingestion.update_document_in_mongodb(
                self.pdf_collection,
                document_id,
                {
                    "processed": True,
                    "document_count": len(documents),
                    "chunk_count": sum(1 for _ in self.data_splitter.split_data(documents))
                }
            )
            
            # Cleanup temp file after successful processing
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Removed temporary file: {temp_file_path}")
            
            return {
                "success": True,
                "document_id": document_id,
                "s3_key": s3_key,
                "vectorized": True,
                "document_count": len(documents)
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF stream {filename}: {str(e)}")
            # Clean up if we got far enough to create records
            if 'document_id' in locals() and 's3_key' in locals():
                self.cleanup_failed_processing(document_id, s3_key, temp_file_path)
            return {
                "success": False,
                "error": str(e)
            }

    def process_multiple_pdfs(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files in parallel
        
        Args:
            files: List of dictionaries with file content and filename
                  Each dict should have 'content' and 'filename' keys
        
        Returns:
            results: List of processing results for each file
        """
        logger.info(f"Processing {len(files)} PDF files")
        
        # Process files in parallel
        futures = []
        for file_info in files:
            future = self.executor.submit(
                self.process_pdf_stream,
                file_info["content"],
                file_info["filename"]
            )
            futures.append(future)
        
        # Collect results
        results = [future.result() for future in futures]
        return results

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
        
    def extract_text_from_pdf_stream(self, file_content: bytes) -> str:
        """
        Extract text from a PDF file content without saving to disk
        
        Args:
            file_content: The binary content of the PDF file
        
        Returns:
            text: The extracted text
        """
        logger.info(f"Extracting text from PDF stream")
        
        try:
            # Create a BytesIO object from the file content
            pdf_stream = io.BytesIO(file_content)
            
            # Use PyPDF2 to extract text
            pdf_reader = PdfReader(pdf_stream)
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
            if text:
                logger.info(f"Successfully extracted {len(text)} characters from PDF stream")
            else:
                logger.warning("No text extracted from PDF stream")
                text = ""
                
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF stream: {str(e)}")
            return ""

    def store_vectors_in_pinecone(self, vector_id: str, vector_values: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Store precomputed vectors in Pinecone
        
        Args:
            vector_id: Unique ID for the vector
            vector_values: The vector embedding
            metadata: Metadata to store with the vector
        
        Returns:
            success: Whether the operation was successful
        """
        logger.info(f"Storing vector {vector_id} in Pinecone")
        
        try:
            # Upsert vector into Pinecone
            self.pinecone_index.upsert(
                vectors=[
                    {"id": vector_id, "values": vector_values, "metadata": metadata}
                ]
            )
            
            logger.info(f"Successfully stored vector {vector_id} in Pinecone")
            return True
        
        except Exception as e:
            logger.error(f"Error storing vector {vector_id} in Pinecone: {str(e)}")
            return False
            
    def create_embeddings(self, text: str) -> List[float]:
        """
        Create embeddings for text using HuggingFace models
        
        Args:
            text: The text to embed
            
        Returns:
            embeddings: List of embedding values
        """
        logger.info("Creating embeddings for text")
        
        try:
            # Generate embeddings
            embedding_vector = self.embeddings.embed_query(text)
            
            logger.info(f"Successfully created embeddings with dimension {len(embedding_vector)}")
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
