#!/usr/bin/env python
"""
Quick reconciliation test script using the improved matcher
"""
import pandas as pd
from data_loader import DataLoader
from data_separator import DataSeparator
from feature_extractor import FeatureExtractor
from matcher import TransactionMatcher
from evaluator import Evaluator
from datetime import datetime
from pathlib import Path

def main():
    print("[STEP 1] Loading data...")
    loader = DataLoader()
    bank_df = loader.load_csv('bank_statements.csv')
    check_df = loader.load_csv('check_register.csv')
    
    if not loader.validate_bank_data(bank_df) or not loader.validate_check_data(check_df):
        print("[ERROR] Validation failed")
        return
    
    print(f"[OK] Loaded {len(bank_df)} bank and {len(check_df)} check transactions")
    
    print("[STEP 2] Cleaning data...")
    bank_df = loader.clean_data(bank_df)
    check_df = loader.clean_data(check_df)
    print(f"[OK] Data cleaned")
    
    print("[STEP 3] Separating training and testing data...")
    separator = DataSeparator()
    training_pairs, testing_candidates = separator.separate_by_amount_uniqueness(bank_df, check_df)
    print(f"[OK] Training pairs: {len(training_pairs)} | Testing: {len(testing_candidates)}")
    
    print("[STEP 4] Initializing embedding model...")
    extractor = FeatureExtractor(model_name='all-MiniLM-L6-v2')
    print(f"[OK] Embedding model ready")
    
    print("[STEP 5] Matching transactions (Hungarian algorithm)...")
    matcher = TransactionMatcher(training_pairs, extractor)
    matched_results = matcher.match_all_transactions(bank_df, check_df, confidence_threshold=0.5)
    print(f"[OK] Matching complete")
    
    print("[STEP 6] Calculating metrics...")
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(matched_results)
    report = evaluator.generate_report(metrics)
    print(f"[OK] Metrics calculated")
    
    print("[STEP 7] Saving results...")
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'matched_transactions_{timestamp}.csv'
    matched_results.to_csv(results_file, index=False, encoding='utf-8')
    print(f"[OK] Results saved: {results_file}")
    
    report_file = output_dir / f'reconciliation_report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] Report saved: {report_file}")
    
    # Display summary
    print("\n" + "="*80)
    print("[SUCCESS] RECONCILIATION COMPLETE!")
    print("="*80)
    print(f"\n[SUMMARY]")
    print(f"   Total Transactions: {metrics['total_transactions']}")
    print(f"   Matched: {metrics['matched_count']}")
    print(f"   Unmatched: {metrics['unmatched_count']}")
    print(f"   Match Rate: {metrics['match_rate']:.1f}%")
    print(f"   Avg Confidence: {metrics['avg_confidence_score']:.4f}")
    print(f"\n[OUTPUT] Files saved to: {output_dir.absolute()}")
    print("\n" + report)

if __name__ == '__main__':
    main()
