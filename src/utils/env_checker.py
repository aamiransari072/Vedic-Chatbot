import os
from typing import List
from dotenv import load_dotenv
from src.logging import logger

# Load environment variables
load_dotenv()

def check_required_env_vars() -> None:
    """
    Check if all required environment variables are set.
    Raises ValueError if any required variable is missing.
    """
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET_NAME",
        "MONGO_URI",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX_NAME",
        "GOOGLE_API_KEY"
    ]
    
    missing_vars: List[str] = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("All required environment variables are set") 