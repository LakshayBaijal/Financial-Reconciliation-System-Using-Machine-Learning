import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class DataSeparator:
    """Separate data into training (unique amounts) and testing (non-unique amounts)"""
    
    @staticmethod
    def separate_by_amount_uniqueness(bank_df: pd.DataFrame, check_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify which amounts are unique (appear once in each file) vs non-unique (appear multiple times)
        
        Logic:
        - TRAINING: Amounts that appear exactly once in bank AND exactly once in check = easy to match
        - TESTING: Amounts that appear multiple times (ambiguous matches) = need ML
        
        Returns:
            training_pairs: DataFrame with unique amount matches (ground truth)
            testing_candidates: DataFrame with non-unique amount candidates
        """
        
        bank_df = bank_df.copy()
        check_df = check_df.copy()
        
        # Convert dates to datetime for calculations
        bank_df['date'] = pd.to_datetime(bank_df['date'])
        check_df['date'] = pd.to_datetime(check_df['date'])
        
        # Count how many times each amount appears in each file
        bank_amount_counts = bank_df['amount'].value_counts()
        check_amount_counts = check_df['amount'].value_counts()
        
        # Amounts that appear exactly ONCE in bank
        bank_unique = set(bank_amount_counts[bank_amount_counts == 1].index)
        
        # Amounts that appear exactly ONCE in check
        check_unique = set(check_amount_counts[check_amount_counts == 1].index)
        
        # TRAINING: Amounts unique in both files (appear 1x bank + 1x check = 2x total)
        training_amounts = bank_unique & check_unique
        logger.info(f"Training amounts (unique in both): {len(training_amounts)}")
        
        # TESTING: Amounts that are non-unique (appear 2+ times in either file)
        all_amounts = set(bank_amount_counts.index) | set(check_amount_counts.index)
        testing_amounts = all_amounts - training_amounts
        logger.info(f"Testing amounts (non-unique): {len(testing_amounts)}")
        
        # Create training pairs
        training_pairs = []
        for amount in training_amounts:
            bank_match = bank_df[bank_df['amount'] == amount]
            check_match = check_df[check_df['amount'] == amount]
            
            if len(bank_match) > 0 and len(check_match) > 0:
                for _, b_row in bank_match.iterrows():
                    for _, c_row in check_match.iterrows():
                        training_pairs.append({
                            'bank_id': b_row.get('transaction_id', ''),
                            'bank_date': b_row['date'],
                            'bank_description': b_row['description'],
                            'amount': amount,
                            'check_id': c_row.get('transaction_id', ''),
                            'check_date': c_row['date'],
                            'check_description': c_row['description'],
                            'date_diff_days': (b_row['date'] - c_row['date']).days,
                            'match_type': 'unique_amount'
                        })
        
        training_df = pd.DataFrame(training_pairs) if training_pairs else pd.DataFrame()
        logger.info(f"Training pairs created: {len(training_df)}")
        
        # Create testing candidates
        testing_candidates = []
        for amount in testing_amounts:
            bank_matches = bank_df[bank_df['amount'] == amount]
            check_matches = check_df[check_df['amount'] == amount]
            
            for _, b_row in bank_matches.iterrows():
                for _, c_row in check_matches.iterrows():
                    testing_candidates.append({
                        'bank_id': b_row.get('transaction_id', ''),
                        'bank_date': b_row['date'],
                        'bank_description': b_row['description'],
                        'amount': amount,
                        'check_id': c_row.get('transaction_id', ''),
                        'check_date': c_row['date'],
                        'check_description': c_row['description'],
                        'date_diff_days': (b_row['date'] - c_row['date']).days,
                        'match_type': 'non_unique_amount'
                    })
        
        testing_df = pd.DataFrame(testing_candidates) if testing_candidates else pd.DataFrame()
        logger.info(f"Testing candidates created: {len(testing_df)}")
        
        return training_df, testing_df