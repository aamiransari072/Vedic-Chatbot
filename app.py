from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import boto3
import pymongo
from bson import ObjectId
from dotenv import load_dotenv
import os

from src.pipeline.pdf_processing_pipeline import PDFProcessingPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.logging import logger
from src.utils.environment import check_env_variables

# Load environment variables and check for required ones
load_dotenv()
check_env_variables()

app = FastAPI(title="Vedic Pedia AI API", description="API for processing PDFs and storing them in vector database")

# Initialize the PDF processing pipeline
pdf_pipeline = PDFProcessingPipeline()

# Initialize the query pipeline
query_pipeline = QueryPipeline()

class PDFResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    s3_key: Optional[str] = None
    vectorized: Optional[bool] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    use_reranking: bool = False
    top_k: Optional[int] = None
    top_k_retrieve: Optional[int] = None
    top_k_rerank: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    success: bool

@app.post("/process-pdfs", response_model=List[PDFResponse])
async def process_pdfs(files: List[UploadFile] = File(...)):
    """
    Process multiple PDF files.
    
    - Upload PDFs to S3 bucket
    - Store metadata in MongoDB
    - Extract text from PDFs
    - Store vectors in Pinecone vector DB
    
    Returns processing results for each file
    """
    logger.info(f"Received request to process {len(files)} PDFs")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Check if all files are PDFs
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
    
    # Process PDFs
    try:
        # Prepare files for processing
        file_list = []
        for file in files:
            content = await file.read()
            file_list.append({
                "content": content,
                "filename": file.filename
            })
        
        # Process PDFs using the pipeline
        results = pdf_pipeline.process_multiple_pdfs(file_list)
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query using the query pipeline.
    
    - Retrieve relevant documents from vector database
    - Use Gemini LLM to generate an answer based on retrieved documents
    
    Returns the answer and source information
    """
    logger.info(f"Received query request: {request.query}")
    
    try:
        if request.use_reranking:
            # Use reranking if requested
            result = query_pipeline.answer_query_with_reranking(
                query=request.query,
                top_k_retrieve=request.top_k_retrieve or 10,
                top_k_rerank=request.top_k_rerank or 5
            )
        else:
            # Standard query processing
            result = query_pipeline.answer_query(
                query=request.query
            )
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to Vedic Pedia AI API"}





