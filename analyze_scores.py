import pandas as pd
import numpy as np
import glob

# Load CSV with proper error handling
csv_files = glob.glob('output/matched_transactions*.csv')
csv_files = [f for f in csv_files if f.endswith('.csv')]
csv_files.sort()

if not csv_files:
    print("[ERROR] No CSV files found in output folder!")
    print("Please run reconciliation first to generate results.")
    exit(1)

latest = csv_files[-1]

df = pd.read_csv(latest)
matched = df[df['match_status'] == 'matched']

print("=== CONFIDENCE DISTRIBUTION ===")
print(f"Total matched: {len(matched)}")
print(f"Min confidence: {matched['confidence_score'].min():.4f}")
print(f"Max confidence: {matched['confidence_score'].max():.4f}")
print(f"Mean confidence: {matched['confidence_score'].mean():.4f}")
print(f"Median confidence: {matched['confidence_score'].median():.4f}")

# Distribution
print(f"\nTotal rows: {len(df)}")
print(f"Matched: {len(matched)} ({100*len(matched)/len(df):.1f}%)")
print(f"Unmatched: {len(df[df['match_status'] == 'unmatched'])} ({100*(len(df)-len(matched))/len(df):.1f}%)")

# Check for duplicate check matches
print("\n=== CHECKING FOR DUPLICATE MATCHES ===")
check_matches = matched[matched['check_id'].notna()]['check_id'].value_counts()
duplicates = check_matches[check_matches > 1]
if len(duplicates) > 0:
    print(f"WARNING: {len(duplicates)} check transactions matched multiple times!")
    print(duplicates.head())
else:
    print("No duplicate matches found (good)")

print("\n=== TEXT SIMILARITY STATS ===")
if 'text_similarity' in matched.columns:
    print(f"Mean text similarity: {matched['text_similarity'].mean():.4f}")
    print(f"Min text similarity: {matched['text_similarity'].min():.4f}")

print("\n=== AMOUNT SIMILARITY STATS ===")
if 'amount_match' in matched.columns:
    print(f"Amount matches: {matched['amount_match'].sum()}")
    print(f"Perfect matches: {(matched['amount_match'] == True).sum()}")

