import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract text and numerical features for matching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with sentence transformer model"""
        try:
            self.model = SentenceTransformer(model_name)
            self.scaler = StandardScaler()
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def normalize_date_diff(self, date_diff_days: int) -> float:
        """Normalize date difference (0-5 days is acceptable)"""
        # Clamp to -5 to +5 days range
        clamped = max(-5, min(5, date_diff_days))
        # Normalize to 0-1 where 0 is max difference (5 days), 1 is same day
        normalized = 1.0 - (abs(clamped) / 5.0)
        return normalized
    
    def normalize_amount(self, amount: float) -> float:
        """
        Normalize amount for feature vector
        
        Handles negative amounts (debits) safely by using absolute value,
        then applies log scale for better distribution across typical ranges.
        """
        # Use absolute value to handle negative amounts (debits)
        abs_amount = abs(amount)
        
        # Apply log scale (works safely with log1p which handles 0)
        log_amount = np.log1p(abs_amount)
        
        # Normalize: assume max log amount ~7 (for ~$1000)
        normalized = log_amount / 7.0
        return min(1.0, max(0.0, normalized))
    
    def extract_features(self, df: pd.DataFrame, text_col: str) -> dict:
        """
        Extract embeddings and numerical features
        
        Args:
            df: DataFrame with transaction data
            text_col: Column name containing description text
        
        Returns:
            dict with embeddings and normalized features
        """
        logger.info(f"Extracting features from {len(df)} rows")
        
        # Get text embeddings
        embeddings = []
        for text in df[text_col]:
            emb = self.get_text_embedding(text)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        # Get numerical features
        date_diffs_norm = df['date_diff_days'].apply(self.normalize_date_diff).values
        amounts_norm = df['amount'].apply(self.normalize_amount).values
        
        return {
            'embeddings': embeddings,
            'date_diffs_norm': date_diffs_norm,
            'amounts_norm': amounts_norm,
            'raw_amounts': df['amount'].values
        }
    
    def create_combined_feature_vector(self, text_emb: np.ndarray, 
                                      date_diff_norm: float, 
                                      amount_norm: float) -> np.ndarray:
        """Combine text embedding with numerical features"""
        # Use text embedding (384 dims for MiniLM) + numerical features
        numerical_features = np.array([date_diff_norm, amount_norm])
        combined = np.concatenate([text_emb, numerical_features])
        return combined
