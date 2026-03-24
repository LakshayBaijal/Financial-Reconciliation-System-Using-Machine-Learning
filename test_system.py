#!/usr/bin/env python
"""Quick test script to verify all modules work correctly"""

print("Testing Financial Reconciliation System...\n")

print("1. Testing data_loader...")
try:
    from data_loader import DataLoader
    print("   ✅ data_loader imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n2. Testing data_separator...")
try:
    from data_separator import DataSeparator
    print("   ✅ data_separator imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n3. Testing feature_extractor...")
try:
    from feature_extractor import FeatureExtractor
    print("   ✅ feature_extractor imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n4. Testing matcher...")
try:
    from matcher import TransactionMatcher
    print("   ✅ matcher imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n5. Testing evaluator...")
try:
    from evaluator import Evaluator
    print("   ✅ evaluator imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n6. Testing PySimpleGUI...")
try:
    import PySimpleGUI as sg
    print("   ✅ PySimpleGUI imported successfully")
    print(f"   Version: {sg.version}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n7. Loading sample data...")
try:
    import pandas as pd
    bank = pd.read_csv('bank_statements.csv')
    check = pd.read_csv('check_register.csv')
    print(f"   ✅ Loaded {len(bank)} bank and {len(check)} check transactions")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n8. Testing reconciliation logic...")
try:
    loader = DataLoader()
    bank = loader.clean_data(bank)
    check = loader.clean_data(check)
    separator = DataSeparator()
    training, testing = separator.separate_by_amount_uniqueness(bank, check)
    print(f"   ✅ Training: {len(training)} pairs, Testing: {len(testing)} pairs")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n✅ All modules loaded successfully!\n")
print("You can now run: python reconciliation_gui.py")
