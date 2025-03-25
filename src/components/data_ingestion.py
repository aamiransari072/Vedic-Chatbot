import os
import uuid
from typing import Dict, Any, Optional, List
import boto3
import pymongo
from bson import ObjectId
from dotenv import load_dotenv
from src.logging import logger

load_dotenv()

class DataIngestionService:
    """
    Service for ingesting data into the system:
    - Uploading files to S3
    - Storing metadata in MongoDB
    """
    
    def __init__(self):
        """Initialize S3 and MongoDB connections"""
        logger.info("Initializing DataIngestionService")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        self.bucket_name = os.environ.get("S3_BUCKET_NAME")
        logger.info(f"S3 client initialized with bucket: {self.bucket_name}")
        
        # Initialize MongoDB client
        self.mongo_client = pymongo.MongoClient(os.environ.get("MONGO_URI"))
        self.db = self.mongo_client["document_db"]
        logger.info("MongoDB connection established")
        
        # Create temp directory for document storage if needed
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Temporary directory created at: {self.temp_dir}")
    
    def upload_file_to_s3(self, file_content: bytes, filename: str, folder: str = "documents") -> str:
        """
        Upload a file to S3
        
        Args:
            file_content: The binary content of the file
            filename: The name of the file
            folder: The folder in the S3 bucket to store the file in
        
        Returns:
            s3_key: The S3 key where the file was uploaded
        """
        logger.info(f"Uploading file {filename} to S3 in folder {folder}")
        
        # Generate a unique ID for the file
        file_id = str(uuid.uuid4())
        s3_key = f"{folder}/{file_id}/{filename}"
        
        # Upload file to S3
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=file_content
        )
        
        logger.info(f"File uploaded to S3 with key: {s3_key}")
        return s3_key
    
    def save_metadata_to_mongodb(self, collection_name: str, metadata: Dict[str, Any]) -> str:
        """
        Save metadata to MongoDB
        
        Args:
            collection_name: The name of the MongoDB collection
            metadata: The metadata to save
        
        Returns:
            document_id: The ID of the document in MongoDB
        """
        logger.info(f"Saving metadata to MongoDB collection: {collection_name}")
        
        # Get the collection
        collection = self.db[collection_name]
        
        # Insert metadata into MongoDB
        result = collection.insert_one(metadata)
        document_id = str(result.inserted_id)
        
        logger.info(f"Metadata saved to MongoDB with ID: {document_id}")
        return document_id
    
    def update_document_in_mongodb(self, collection_name: str, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a document in MongoDB
        
        Args:
            collection_name: The name of the MongoDB collection
            document_id: The ID of the document to update
            updates: The updates to apply
        
        Returns:
            success: Whether the update was successful
        """
        logger.info(f"Updating document {document_id} in collection {collection_name}")
        
        # Get the collection
        collection = self.db[collection_name]
        
        try:
            # Update the document
            result = collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": updates}
            )
            
            success = result.modified_count > 0
            logger.info(f"Document update {'successful' if success else 'failed'}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    def get_document_from_mongodb(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from MongoDB by ID
        
        Args:
            collection_name: The name of the MongoDB collection
            document_id: The ID of the document to get
        
        Returns:
            document: The document, or None if not found
        """
        logger.info(f"Getting document {document_id} from collection {collection_name}")
        
        # Get the collection
        collection = self.db[collection_name]
        
        try:
            # Get the document
            document = collection.find_one({"_id": ObjectId(document_id)})
            
            if document:
                # Convert ObjectId to string for serialization
                document["_id"] = str(document["_id"])
                logger.info(f"Document found: {document_id}")
            else:
                logger.warning(f"Document not found: {document_id}")
                
            return document
            
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            return None
    
    def save_temporary_file(self, file_content: bytes, filename: str) -> str:
        """
        Save a file temporarily for processing
        
        Args:
            file_content: The binary content of the file
            filename: The name of the file
        
        Returns:
            file_path: The path where the file was saved
        """
        file_path = os.path.join(self.temp_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"File saved temporarily at: {file_path}")
        return file_path
    
    def cleanup_temporary_file(self, file_path: str) -> None:
        """
        Clean up a temporary file
        
        Args:
            file_path: The path to the file to delete
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
    
    def download_file_from_s3(self, s3_key: str, local_path: Optional[str] = None) -> str:
        """
        Download a file from S3
        
        Args:
            s3_key: The S3 key of the file to download
            local_path: The local path to save the file to (optional)
        
        Returns:
            file_path: The path where the file was saved
        """
        logger.info(f"Downloading file from S3: {s3_key}")
        
        if local_path is None:
            # Extract the filename from the S3 key
            filename = s3_key.split("/")[-1]
            local_path = os.path.join(self.temp_dir, filename)
        
        # Download the file
        self.s3_client.download_file(
            Bucket=self.bucket_name,
            Key=s3_key,
            Filename=local_path
        )
        
        logger.info(f"File downloaded to: {local_path}")
        return local_path
