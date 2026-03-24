"""
Run this ONCE to pre-train and save embeddings.
After running, commit the output files to version control.
"""

import pandas as pd
import pickle
import json
from pathlib import Path
from data_loader import DataLoader
from data_separator import DataSeparator
from feature_extractor import FeatureExtractor

def train_model():
    print("🚀 Starting one-time training...")
    print("="*80)
    
    # Load your provided data
    bank_file = 'bank_statements.csv'
    check_file = 'check_register.csv'
    
    print(f"\n📥 Loading {bank_file}...")
    bank_df = pd.read_csv(bank_file)
    print(f"   Loaded {len(bank_df)} transactions")
    
    print(f"\n📥 Loading {check_file}...")
    check_df = pd.read_csv(check_file)
    print(f"   Loaded {len(check_df)} transactions")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    loader = DataLoader()
    bank_df = loader.clean_data(bank_df)
    check_df = loader.clean_data(check_df)
    print(f"   Bank: {len(bank_df)} | Check: {len(check_df)}")
    
    # Separate
    print("\n🔄 Separating training and testing data...")
    separator = DataSeparator()
    training_pairs, testing_candidates = separator.separate_by_amount_uniqueness(bank_df, check_df)
    print(f"   Training pairs: {len(training_pairs)}")
    print(f"   Testing candidates: {len(testing_candidates)}")
    
    # Extract embeddings
    print("\n🧠 Generating embeddings (this may take 2-3 minutes)...")
    print("   Loading sentence-transformers model...")
    extractor = FeatureExtractor(model_name='all-MiniLM-L6-v2')
    
    # Generate embeddings for all descriptions in training data
    if len(training_pairs) > 0:
        bank_descriptions = training_pairs['bank_description'].unique()
        check_descriptions = training_pairs['check_description'].unique()
        
        print(f"   Generating {len(bank_descriptions)} bank embeddings...")
        bank_embeddings = {}
        for i, desc in enumerate(bank_descriptions):
            bank_embeddings[desc] = extractor.get_text_embedding(desc)
            if (i + 1) % 50 == 0:
                print(f"      Progress: {i+1}/{len(bank_descriptions)}")
        
        print(f"   Generating {len(check_descriptions)} check embeddings...")
        check_embeddings = {}
        for i, desc in enumerate(check_descriptions):
            check_embeddings[desc] = extractor.get_text_embedding(desc)
            if (i + 1) % 50 == 0:
                print(f"      Progress: {i+1}/{len(check_descriptions)}")
    else:
        bank_embeddings = {}
        check_embeddings = {}
    
    # Save embeddings
    print("\n💾 Saving embeddings...")
    
    embeddings_data = {
        'bank_embeddings': bank_embeddings,
        'check_embeddings': check_embeddings,
        'training_pairs': training_pairs.to_dict('records') if len(training_pairs) > 0 else [],
        'testing_candidates': testing_candidates.to_dict('records') if len(testing_candidates) > 0 else []
    }
    
    output_file = 'training_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print(f"   ✅ Embeddings saved to {output_file}")
    
    # Save training info
    info = {
        'total_bank_descriptions': len(bank_embeddings),
        'total_check_descriptions': len(check_embeddings),
        'training_pairs_count': len(training_pairs),
        'testing_candidates_count': len(testing_candidates),
        'total_bank_transactions': len(bank_df),
        'total_check_transactions': len(check_df),
    }
    
    with open('training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✅ Training info saved to training_info.json")
    
    # Display summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Total Bank Transactions: {info['total_bank_transactions']}")
    print(f"   Total Check Transactions: {info['total_check_transactions']}")
    print(f"   Training Pairs: {info['training_pairs_count']}")
    print(f"   Testing Candidates: {info['testing_candidates_count']}")
    print(f"   Bank Embeddings: {info['total_bank_descriptions']}")
    print(f"   Check Embeddings: {info['total_check_descriptions']}")
    print(f"\n✅ You can now create the .EXE file!")
    print("="*80)

if __name__ == '__main__':
    train_model()
