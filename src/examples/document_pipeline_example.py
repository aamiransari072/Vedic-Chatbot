from src.components.document_processing import DocumentProcessor
from src.components.document_extraction import DocumentExtractor
import logging
import json
import os
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_extracted_text(extracted_docs: List[Dict], output_dir: str = "extracted_texts"):
    """
    Save extracted text from documents to individual text files
    
    Args:
        extracted_docs: List of document dictionaries with extracted_text field
        output_dir: Directory to save the extracted texts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving extracted texts to directory: {output_dir}")
    
    saved_count = 0
    for doc in extracted_docs:
        if "extracted_text" in doc and doc["extracted_text"]:
            # Create a filename based on document title or ID
            title = doc.get("title", "")
            doc_id = str(doc.get("_id", ""))
            
            if title:
                # Replace invalid filename characters
                filename = "".join(c if c.isalnum() or c in [' ', '.', '_', '-'] else '_' for c in title)
                filename = f"{filename}_{doc_id[:8]}.txt"
            else:
                filename = f"document_{doc_id}.txt"
            
            # Save to file
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc["extracted_text"])
            
            saved_count += 1
            logger.info(f"Saved extracted text to: {output_path}")
    
    logger.info(f"Saved extracted text for {saved_count} documents")
    return saved_count

def document_processing_pipeline():
    """
    Full document processing pipeline example:
    1. Retrieve documents from MongoDB
    2. Download documents from S3
    3. Extract text from downloaded documents
    4. Save extracted text
    5. Clean up temporary files
    """
    logger.info("Starting document processing pipeline")
    
    # Step 1 & 2: Initialize document processor and retrieve/download documents
    doc_processor = DocumentProcessor()
    
    # Retrieve and download documents (adjust collection name and query as needed)
    collection_name = "documents"
    query = {"document_type": "vedic_text"}  # Example query
    limit = 10  # Example limit
    
    logger.info(f"Retrieving documents from collection '{collection_name}' with query: {query}")
    documents = doc_processor.get_and_download_documents(collection_name, query, limit)
    
    if not documents:
        logger.warning("No documents found or downloaded")
        return
    
    logger.info(f"Retrieved and downloaded {len(documents)} documents")
    
    # Step 3: Extract text from downloaded documents
    doc_extractor = DocumentExtractor(temp_dir=doc_processor.temp_dir)
    extracted_docs = doc_extractor.extract_from_document_list(documents)
    
    # Step 4: Save extracted text to files
    save_extracted_text(extracted_docs, "extracted_texts")
    
    # Step 5: Clean up temporary files (optional)
    logger.info("Cleaning up temporary files")
    doc_processor.clear_temp_directory()
    
    logger.info("Document processing pipeline completed")
    
    # Return the extracted documents for further processing if needed
    return extracted_docs

def print_document_summary(documents: List[Dict]):
    """Print a summary of the processed documents"""
    if not documents:
        logger.warning("No documents to summarize")
        return
    
    print("\n" + "="*50)
    print(f"DOCUMENT PROCESSING SUMMARY ({len(documents)} documents)")
    print("="*50)
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}: {doc.get('title', 'Untitled')}")
        print(f"ID: {doc.get('_id', 'Unknown')}")
        
        if "s3_key" in doc:
            print(f"S3 Key: {doc['s3_key']}")
        
        if "local_path" in doc:
            print(f"Downloaded to: {doc['local_path']}")
        
        if "extracted_text" in doc:
            text_preview = doc["extracted_text"][:150] + "..." if len(doc["extracted_text"]) > 150 else doc["extracted_text"]
            print(f"Extracted Text ({len(doc['extracted_text'])} chars): {text_preview}")
        else:
            print("No text extracted")
        
        print("-"*40)

if __name__ == "__main__":
    # Run the full pipeline
    processed_docs = document_processing_pipeline()
    
    # Print summary of processed documents
    if processed_docs:
        print_document_summary(processed_docs) 