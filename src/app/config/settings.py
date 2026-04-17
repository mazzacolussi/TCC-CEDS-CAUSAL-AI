from pathlib import Path
from dotenv import load_dotenv
from app.data import get_features
from app.data.utils import find_specific_variables

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Useful variables
date_cols = find_specific_variables(
    get_features('features.yaml'), 
    "type", 
    specific_value = 'datetime'
)
