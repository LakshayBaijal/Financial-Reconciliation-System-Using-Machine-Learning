import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class TransactionMatcher:
    """Match transactions using embeddings and similarity scoring"""
    
    def __init__(self, training_pairs: pd.DataFrame, feature_extractor):
        """
        Initialize matcher with training data
        
        Args:
            training_pairs: Ground truth matches from unique amounts
            feature_extractor: FeatureExtractor instance
        """
        self.training_pairs = training_pairs
        self.feature_extractor = feature_extractor
        self.logger = logging.getLogger(__name__)
        
        # Pre-compute training embeddings
        self._compute_training_embeddings()
    
    def _compute_training_embeddings(self):
        """Pre-compute embeddings for all training descriptions"""
        if len(self.training_pairs) == 0:
            self.bank_desc_embeddings = {}
            self.check_desc_embeddings = {}
            return
            
        bank_descs = self.training_pairs['bank_description'].unique()
        check_descs = self.training_pairs['check_description'].unique()
        
        self.bank_desc_embeddings = {
            desc: self.feature_extractor.get_text_embedding(desc)
            for desc in bank_descs
        }
        
        self.check_desc_embeddings = {
            desc: self.feature_extractor.get_text_embedding(desc)
            for desc in check_descs
        }
        
        self.logger.info(f"Computed embeddings for {len(bank_descs)} bank + {len(check_descs)} check descriptions")
    
    def match_all_transactions(self, bank_df: pd.DataFrame, check_df: pd.DataFrame, 
                              confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Match all transactions using optimal assignment algorithm
        
        This uses the Hungarian algorithm to find the best one-to-one matching
        that maximizes overall confidence scores.
        
        Returns:
            Complete matched results with confidence scores
        """
        n_bank = len(bank_df)
        n_check = len(check_df)
        
        # Build cost matrix: higher is better (we'll minimize negative)
        # Shape: (n_bank, n_check)
        cost_matrix = np.zeros((n_bank, n_check))
        
        # Pre-compute check embeddings
        check_embeddings = {}
        for idx, row in check_df.iterrows():
            if row['description'] not in check_embeddings:
                check_embeddings[row['description']] = self.feature_extractor.get_text_embedding(row['description'])
        
        # Compute all scores
        for i, (_, bank_row) in enumerate(bank_df.iterrows()):
            bank_emb = self.feature_extractor.get_text_embedding(bank_row['description'])
            
            for j, (_, check_row) in enumerate(check_df.iterrows()):
                check_emb = check_embeddings.get(check_row['description'])
                if check_emb is None:
                    check_emb = self.feature_extractor.get_text_embedding(check_row['description'])
                
                # Text similarity (cosine)
                text_similarity = cosine_similarity([bank_emb], [check_emb])[0][0]
                
                # Amount similarity (prefer exact or very close matches)
                amount_diff = abs(bank_row['amount'] - check_row['amount'])
                amount_similarity = 1.0 if amount_diff < 0.01 else (1.0 / (1.0 + amount_diff))
                
                # Date proximity
                date_diff = (bank_row['date'] - check_row['date']).days
                date_score = self.feature_extractor.normalize_date_diff(date_diff)
                
                # Combined scoring (higher is better)
                # 50% text similarity, 30% amount similarity, 20% date score
                combined_score = (0.5 * text_similarity) + (0.3 * amount_similarity) + (0.2 * date_score)
                
                # Store as negative for minimization (linear_sum_assignment minimizes cost)
                cost_matrix[i, j] = -combined_score
        
        # Use Hungarian algorithm for optimal assignment
        bank_indices, check_indices = linear_sum_assignment(cost_matrix)
        
        # Build matched set from optimal assignment
        matched_map = {}  # Maps bank_idx -> (check_idx, score)
        for b_idx, c_idx in zip(bank_indices, check_indices):
            score = -cost_matrix[b_idx, c_idx]  # Convert back to positive
            if score >= confidence_threshold:  # Only keep matches above threshold
                matched_map[b_idx] = (c_idx, score)
        
        # Build results
        all_results = []
        
        for b_idx, (_, bank_row) in enumerate(bank_df.iterrows()):
            # Check if this bank transaction has an optimal match from Hungarian algorithm
            if b_idx in matched_map:
                c_idx, best_score = matched_map[b_idx]
                check_row = check_df.iloc[c_idx]
                
                # Recalculate components for result
                bank_emb = self.feature_extractor.get_text_embedding(bank_row['description'])
                check_emb = check_embeddings.get(check_row['description'])
                if check_emb is None:
                    check_emb = self.feature_extractor.get_text_embedding(check_row['description'])
                
                text_sim = cosine_similarity([bank_emb], [check_emb])[0][0]
                amt_diff = abs(bank_row['amount'] - check_row['amount'])
                amt_sim = 1.0 if amt_diff < 0.01 else (1.0 / (1.0 + amt_diff))
                date_diff = (bank_row['date'] - check_row['date']).days
                date_score = self.feature_extractor.normalize_date_diff(date_diff)
                
                all_results.append({
                    'bank_id': bank_row.get('transaction_id', ''),
                    'bank_date': bank_row['date'],
                    'bank_description': bank_row['description'],
                    'bank_amount': bank_row['amount'],
                    'check_id': check_row.get('transaction_id', ''),
                    'check_date': check_row['date'],
                    'check_description': check_row['description'],
                    'check_amount': check_row['amount'],
                    'amount_match': amt_diff < 0.01,
                    'date_diff_days': date_diff,
                    'text_similarity': text_sim,
                    'amount_similarity': amt_sim,
                    'date_score': date_score,
                    'confidence_score': min(1.0, best_score),
                    'match_status': 'matched'
                })
            else:
                # Unmatched bank transaction
                all_results.append({
                    'bank_id': bank_row.get('transaction_id', ''),
                    'bank_date': bank_row['date'],
                    'bank_description': bank_row['description'],
                    'bank_amount': bank_row['amount'],
                    'check_id': None,
                    'check_date': None,
                    'check_description': None,
                    'check_amount': None,
                    'amount_match': False,
                    'date_diff_days': None,
                    'text_similarity': None,
                    'amount_similarity': None,
                    'date_score': None,
                    'confidence_score': 0.0,
                    'match_status': 'unmatched'
                })
        
        return pd.DataFrame(all_results)
