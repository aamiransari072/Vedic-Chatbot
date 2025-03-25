import os
import sys
from pathlib import Path
import argparse
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

from src.pipeline.query_pipeline import QueryPipeline
from src.utils.environment import check_env_variables, get_env_variable
from src.logging import logger
from dotenv import load_dotenv

def test_query_pipeline(query, use_reranking=False):
    """
    Test the query pipeline with a user query
    
    Args:
        query: The user's query string
        use_reranking: Whether to use reranking of retrieved documents
    """
    # Load environment variables
    load_dotenv()
    
    try:
        # Check environment variables
        check_env_variables()
        
        # Initialize the pipeline
        pipeline = QueryPipeline()
        
        # Process the query
        if use_reranking:
            logger.info("Using retrieval with reranking")
            result = pipeline.answer_query_with_reranking(query)
        else:
            logger.info("Using standard retrieval")
            result = pipeline.answer_query(query)
        
        # Print the result
        logger.info(f"Query: {query}")
        logger.info(f"Answer: {result['answer']}")
        
        # Print sources
        if result["sources"]:
            logger.info(f"Sources ({len(result['sources'])}):")
            for i, source in enumerate(result["sources"]):
                logger.info(f"  Source {i+1}: {source.get('filename', 'Unknown')} (ID: {source.get('document_id', 'Unknown')})")
        else:
            logger.info("No sources found")
        
        return result
    
    except Exception as e:
        logger.error(f"Error testing query pipeline: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "success": False
        }

def save_result_to_file(result, output_file):
    """
    Save the query result to a JSON file
    
    Args:
        result: The query result
        output_file: Path to the output file
    """
    try:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving result to file: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the query pipeline")
    parser.add_argument("query", help="The query to process")
    parser.add_argument("--rerank", action="store_true", help="Use reranking of retrieved documents")
    parser.add_argument("--output", help="Path to save the result as JSON")
    args = parser.parse_args()
    
    # Test the pipeline
    result = test_query_pipeline(args.query, args.rerank)
    
    # Save result if output file specified
    if args.output:
        save_result_to_file(result, args.output) 