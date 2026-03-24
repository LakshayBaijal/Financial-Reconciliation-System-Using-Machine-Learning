import pandas as pd
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate bank statements and check register CSVs"""
    
    @staticmethod
    def load_csv(file) -> pd.DataFrame:
        """Load CSV file and return DataFrame"""
        try:
            if isinstance(file, str):
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(file)
            logger.info(f"Loaded file with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    @staticmethod
    def validate_bank_data(df: pd.DataFrame) -> bool:
        """Validate bank statement has required columns"""
        required_cols = ['date', 'description', 'amount', 'type']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Bank data missing columns: {missing}")
            return False
        return True
    
    @staticmethod
    def validate_check_data(df: pd.DataFrame) -> bool:
        """Validate check register has required columns"""
        required_cols = ['date', 'description', 'amount', 'type']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Check data missing columns: {missing}")
            return False
        return True
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean data: remove duplicates, handle nulls, standardize types"""
        df = df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert amount to float
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows with null values in critical columns
        df = df.dropna(subset=['date', 'description', 'amount'])
        
        # Standardize description to lowercase
        df['description'] = df['description'].str.lower().str.strip()
        
        logger.info(f"Data cleaned, {len(df)} rows remaining")
        return df
