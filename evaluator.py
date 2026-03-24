import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluate matching performance"""
    
    @staticmethod
    def calculate_metrics(matched_df: pd.DataFrame) -> dict:
        """
        Calculate comprehensive metrics using transaction_id as ground truth.
        
        IMPORTANT: The ground truth is simple:
        - Every bank transaction MUST have exactly one check transaction
        - Bank B0047 should match Check R0047 (by numeric ID)
        - Any mismatch is a FAILURE of the algorithm
        
        This creates a bipartite matching problem where perfect score = all correct matches
        """
        total_bank_txns = len(matched_df['bank_id'].unique())
        matched_count = (matched_df['match_status'] == 'matched').sum()
        unmatched_count = (matched_df['match_status'] == 'unmatched').sum()
        
        # Confidence distribution
        high_confidence = (matched_df['confidence_score'] >= 0.85).sum()
        medium_confidence = ((matched_df['confidence_score'] >= 0.70) & 
                            (matched_df['confidence_score'] < 0.85)).sum()
        low_confidence = ((matched_df['confidence_score'] > 0.0) & 
                         (matched_df['confidence_score'] < 0.70)).sum()
        
        # Confidence statistics (for matched only)
        matched_mask = matched_df['match_status'] == 'matched'
        avg_conf = matched_df[matched_mask]['confidence_score'].mean() if matched_mask.any() else 0.0
        min_conf = matched_df[matched_mask]['confidence_score'].min() if matched_mask.any() else 0.0
        max_conf = matched_df[matched_mask]['confidence_score'].max() if matched_mask.any() else 0.0
        
        # ===== GROUND TRUTH EVALUATION =====
        # Extract numeric parts: B0047 → 0047, R0047 → 0047
        def extract_id_number(transaction_id):
            """Extract numeric portion from transaction_id (B0047 → 0047)"""
            if pd.isna(transaction_id):
                return None
            return str(transaction_id)[1:] if len(str(transaction_id)) > 1 else str(transaction_id)
        
        matched_df['bank_id_num'] = matched_df['bank_id'].apply(extract_id_number)
        matched_df['check_id_num'] = matched_df['check_id'].apply(extract_id_number)
        
        # Ground truth: bank_id_num == check_id_num means CORRECT match
        # In a perfect bipartite matching, EVERY bank transaction should match one check transaction
        matched_df['ground_truth_correct'] = (
            (matched_df['bank_id_num'] == matched_df['check_id_num']) & 
            matched_df['check_id_num'].notna()
        ).astype(int)
        
        # Predictions: our match_status == 'matched'
        matched_df['predicted_match'] = (matched_df['match_status'] == 'matched').astype(int)
        
        # For bipartite matching: 
        # TP = correct matches we made
        # FP = incorrect matches we made  
        # FN = matches we failed to make (all unmatched transactions, since every transaction SHOULD match)
        
        tp = int(((matched_df['ground_truth_correct'] == 1) & (matched_df['predicted_match'] == 1)).sum())
        fp = int(((matched_df['ground_truth_correct'] == 0) & (matched_df['predicted_match'] == 1)).sum())
        fn = unmatched_count  # Every unmatched transaction is a false negative
        
        # Calculate precision, recall, F1 safely
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            # Counts
            'total_transactions': total_bank_txns,
            'matched_count': matched_count,
            'unmatched_count': unmatched_count,
            'match_rate': (matched_count / total_bank_txns * 100) if total_bank_txns > 0 else 0,
            
            # Confidence distribution
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence,
            
            # Confidence statistics
            'avg_confidence_score': avg_conf,
            'min_confidence_score': min_conf,
            'max_confidence_score': max_conf,
            
            # Proper evaluation metrics using ground truth
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }
        
        return metrics
    
    @staticmethod
    def generate_report(metrics: dict) -> str:
        """Generate comprehensive text report with honest evaluation"""
        report = f"""
========================================================
   FINANCIAL RECONCILIATION REPORT
========================================================

GROUND TRUTH SETUP
-------------------------------------------------------
Bank transactions (B0001-B0308) must match Check 
transactions (R0001-R0308) by numeric ID.
This is a perfect one-to-one matching problem.

OVERALL PERFORMANCE
-------------------------------------------------------
Total Bank Transactions:      {metrics['total_transactions']}
Successfully Matched:         {metrics['matched_count']}
Failed to Match:              {metrics['unmatched_count']}
Match Rate:                   {metrics['match_rate']:.2f}%

CONFIDENCE DISTRIBUTION
-------------------------------------------------------
High Confidence (>= 0.85):    {metrics['high_confidence_count']}
Medium Confidence (0.70-0.85):{metrics['medium_confidence_count']}
Low Confidence (0.00-0.70):   {metrics['low_confidence_count']}

CONFIDENCE STATISTICS (Matched Transactions Only)
-------------------------------------------------------
Average Confidence:           {metrics['avg_confidence_score']:.4f}
Minimum Confidence:           {metrics['min_confidence_score']:.4f}
Maximum Confidence:           {metrics['max_confidence_score']:.4f}

EVALUATION METRICS (Using Transaction ID Ground Truth)
-------------------------------------------------------
Ground Truth Rule: B0047 should match R0047 (by numeric ID)

True Positives (Correct matches made):     {metrics['true_positives']}
False Positives (Incorrect matches made):  {metrics['false_positives']}
False Negatives (Correct matches missed):  {metrics['false_negatives']}

Precision = TP / (TP + FP) = {metrics['true_positives']} / ({metrics['true_positives']} + {metrics['false_positives']})
          = {metrics['precision']:.4f}
          
Recall    = TP / (TP + FN) = {metrics['true_positives']} / ({metrics['true_positives']} + {metrics['false_negatives']})
          = {metrics['recall']:.4f}
          
F1 Score  = 2 * (Precision × Recall) / (Precision + Recall)
          = {metrics['f1_score']:.4f}

INTERPRETATION
-------------------------------------------------------
Precision: {metrics['precision']:.4f} - Of the matches made, this % were correct
Recall:    {metrics['recall']:.4f} - Of all correct matches possible, we found this %
F1 Score:  {metrics['f1_score']:.4f} - Harmonic mean balancing precision and recall

========================================================
        """
        return report
