# config.py
import os
import logging
import warnings

# ignore unnecessary warnings
warnings.filterwarnings("ignore")

# Log Configuration
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Environment variable settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
