import os
from dotenv import load_dotenv
from src.logging import logger

# Load environment variables
load_dotenv()

REQUIRED_ENV_VARS = [
    "MONGO_URI",
    "PINECONE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_BUCKET_NAME"
]

def check_env_variables():
    """
    Check that all required environment variables are set
    
    Raises:
        ValueError: If any required environment variable is missing
    """
    missing_vars = []
    
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        error_message = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info("All required environment variables are set")
    
    # Log optional variables and their default values
    if not os.environ.get("PINECONE_ENVIRONMENT"):
        logger.info("PINECONE_ENVIRONMENT not set, using default: gcp-starter")
    
    if not os.environ.get("PINECONE_INDEX_NAME"):
        logger.info("PINECONE_INDEX_NAME not set, using default: pdf-vectors")
    
    if not os.environ.get("AWS_REGION"):
        logger.info("AWS_REGION not set, using default: us-east-1")

def get_env_variable(name, default=None):
    """
    Get an environment variable or return a default value
    
    Args:
        name: The name of the environment variable
        default: The default value to return if the variable is not set
    
    Returns:
        The value of the environment variable or the default value
    """
    value = os.environ.get(name, default)
    
    if value is None:
        logger.warning(f"Environment variable {name} not set and no default provided")
    
    return value 