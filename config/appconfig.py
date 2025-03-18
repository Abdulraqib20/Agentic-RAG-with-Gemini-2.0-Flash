import os
import re
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent

from qdrant_client import QdrantClient

# qdrant_client = QdrantClient(
#     url="https://c1733a44-5b62-43b6-ac36-8e802040a9aa.us-east4-0.gcp.cloud.qdrant.io:6333", 
#     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.W8FpW3L5iV93ZC3KHaEQ1fKXpmbExZAKVGvIJFqW7AA",
# )

# print(qdrant_client.get_collections())


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).resolve().parent.parent / 'logs/config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # Path configuration for .env located at the project root
    env_path = project_root / '.env'
    
    if not env_path.exists():
        logger.critical(f"Missing .env file at {env_path}")
        sys.exit(1)

    load_dotenv(env_path)

    # Required environment variables
    REQUIRED_VARS = [
        'QDRANT_LOCATION',
        'QDRANT_URL',
        'QDRANT_API_KEY',
        'GROQ_API_KEY',
        'GOOGLE_API_KEY',
        'EXAAI_API_KEY'
    ]

    # Load environment variables
    config = {var: os.getenv(var) for var in REQUIRED_VARS}

    missing_vars = [var for var in REQUIRED_VARS if config[var] is None]
    if missing_vars:
        logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Export variables
    QDRANT_LOCATION = config['QDRANT_LOCATION']
    QDRANT_URL = config['QDRANT_URL']
    QDRANT_API_KEY = config['QDRANT_API_KEY']
    GROQ_API_KEY = config['GROQ_API_KEY']
    GOOGLE_API_KEY = config['GOOGLE_API_KEY']
    EXAAI_API_KEY = config['EXAAI_API_KEY']

    # Secure logging for sensitive variables
    sensitive_vars = ['QDRANT_LOCATION', 'QDRANT_URL', 'QDRANT_API_KEY', 
                      'GROQ_API_KEY', 'GOOGLE_API_KEY', 'EXAAI_API_KEY']
    for var in REQUIRED_VARS:
        value = locals().get(var, '')
        logged_value = f"{value[:2]}****{value[-2:]}" if var in sensitive_vars and len(value) > 4 else str(value)
        logger.info(f"{var}: {logged_value}")

    logger.info("Configuration loaded successfully")

except Exception as e:
    logger.critical(f"Configuration initialization failed: {str(e)}")
    sys.exit(1)

__all__ = REQUIRED_VARS
