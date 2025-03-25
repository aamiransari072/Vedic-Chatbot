import os
import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

from src.pipeline.pdf_processing_pipeline import PDFProcessingPipeline
from src.utils.environment import check_env_variables, get_env_variable
from src.logging import logger
from dotenv import load_dotenv

def test_pdf_pipeline(pdf_path):
    """
    Test the PDF processing pipeline with a single PDF file
    
    Args:
        pdf_path: Path to the PDF file to process
    """
    # Load environment variables
    load_dotenv()
    
    try:
        # Check environment variables
        check_env_variables()
        
        # Initialize the pipeline
        pipeline = PDFProcessingPipeline()
        
        # Get the PDF file path
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return
        
        # Read the PDF file
        with open(pdf_file, "rb") as f:
            content = f.read()
        
        # Process the PDF
        result = pipeline.process_pdf(content, pdf_file.name)
        
        # Print the result
        logger.info(f"Processing result: {result}")
        
        if result["success"]:
            logger.info(f"PDF processed successfully with document ID: {result['document_id']}")
            logger.info(f"S3 key: {result['s3_key']}")
            logger.info(f"Vectorized: {result['vectorized']}")
        else:
            logger.error(f"PDF processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Error testing PDF pipeline: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the PDF processing pipeline")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    args = parser.parse_args()
    
    # Test the pipeline
    test_pdf_pipeline(args.pdf_path) 