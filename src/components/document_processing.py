import boto3
import pymongo
import os
import logging
from bson import ObjectId
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.logging import logger

load_dotenv()



class DocumentProcessor:
    """
    A component for retrieving document information from MongoDB and downloading
    documents from S3 bucket.
    """
    
    def __init__(self):
        """Initialize MongoDB and S3 connections"""
        logger.info("Initializing DocumentProcessor")
        # MongoDB connection
        self.mongo_client = pymongo.MongoClient(os.environ.get("MONGO_URI"))
        self.db = None  # Will be set when collection is specified
        logger.info("MongoDB connection established")
        
        # S3 connection
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        self.bucket_name = os.environ.get("S3_BUCKET_NAME")
        logger.info(f"S3 client initialized with bucket: {self.bucket_name}")
        
        # Create temp directory for document storage
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Temporary directory created at: {self.temp_dir}")
    
    def get_documents_from_collection(self, collection_name: str, query: Dict = None, limit: int = 100) -> List[Dict]:
        """
        Retrieve document information from MongoDB collection
        
        Args:
            collection_name: The name of the MongoDB collection
            query: MongoDB query filter (default: None, which retrieves all documents)
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of document records from MongoDB
        """
        if query is None:
            query = {}
        
        logger.info(f"Retrieving documents from collection '{collection_name}' with query: {query}")
            
        self.db = self.mongo_client.get_database()
        collection = self.db[collection_name]
        
        cursor = collection.find(query).limit(limit)
        documents = list(cursor)
        logger.info(f"Retrieved {len(documents)} documents from collection '{collection_name}'")
        return documents
    
    def get_document_by_id(self, collection_name: str, document_id: str) -> Optional[Dict]:
        """
        Retrieve a single document by its ID
        
        Args:
            collection_name: The name of the MongoDB collection
            document_id: The ObjectId of the document as a string
            
        Returns:
            Document record from MongoDB or None if not found
        """
        logger.info(f"Retrieving document with ID '{document_id}' from collection '{collection_name}'")
        
        self.db = self.mongo_client.get_database()
        collection = self.db[collection_name]
        
        try:
            document = collection.find_one({"_id": ObjectId(document_id)})
            if document:
                logger.info(f"Document found: {document.get('title', 'Untitled')}")
            else:
                logger.info(f"Document with ID '{document_id}' not found")
            return document
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None
    
    def download_document_from_s3(self, s3_key: str, local_path: str = None) -> str:
        """
        Download a document from S3 bucket
        
        Args:
            s3_key: The S3 object key of the document
            local_path: Local path to save the document (default: temp file)
            
        Returns:
            Path to the downloaded document
        """
        logger.info(f"Downloading document from S3 with key: {s3_key}")
        
        if local_path is None:
            # Use the temp directory
            filename = s3_key.split("/")[-1]
            local_path = os.path.join(self.temp_dir, filename)
            
        try:
            self.s3_client.download_file(
                self.bucket_name, 
                s3_key, 
                local_path
            )
            logger.info(f"Document downloaded successfully to: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading document from S3: {e}")
            return None
    
    def get_and_download_documents(self, collection_name: str, query: Dict = None, limit: int = 100) -> List[Dict]:
        """
        Retrieve documents from MongoDB and download them from S3
        
        Args:
            collection_name: The name of the MongoDB collection
            query: MongoDB query filter
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of documents with added local_path field pointing to downloaded files
        """
        logger.info(f"Retrieving and downloading documents from collection '{collection_name}'")
        
        documents = self.get_documents_from_collection(collection_name, query, limit)
        
        download_count = 0
        for doc in documents:
            if "s3_key" in doc:
                local_path = self.download_document_from_s3(doc["s3_key"])
                doc["local_path"] = local_path
                if local_path:
                    download_count += 1
        
        logger.info(f"Successfully downloaded {download_count} out of {len(documents)} documents")
        return documents
    
    def clear_temp_directory(self):
        """
        Clear all files from the temporary directory
        """
        logger.info("Clearing temporary directory")
        
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.info(f"Deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
        
        logger.info("Temporary directory cleared")
