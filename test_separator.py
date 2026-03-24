import pandas as pd
from data_loader import DataLoader
from data_separator import DataSeparator

loader = DataLoader()
bank_raw = pd.read_csv('bank_statements.csv')
check_raw = pd.read_csv('check_register.csv')

bank = loader.clean_data(bank_raw)
check = loader.clean_data(check_raw)

print("Before separator:")
print(f"Bank: {len(bank)} rows")
print(f"Check: {len(check)} rows")
print()

# Test the separator
separator = DataSeparator()
training_pairs, testing_candidates = separator.separate_by_amount_uniqueness(bank, check)

print(f"Training pairs: {len(training_pairs)}")
print(f"Testing candidates: {len(testing_candidates)}")
print()

if len(training_pairs) > 0:
    print("Sample training pairs:")
    print(training_pairs[['bank_description', 'amount', 'check_description']].head(10))
else:
    print("NO TRAINING PAIRS FOUND!")
    print("\nDEBUGGING:")
    
    # Manual check
    bank_amounts = bank['amount'].values
    check_amounts = check['amount'].values
    
    all_amounts = list(bank_amounts) + list(check_amounts)
    print(f"Total amount values: {len(all_amounts)}")
    
    from collections import Counter
    amount_counts = Counter(all_amounts)
    
    unique_once = [amt for amt, count in amount_counts.items() if count == 1]
    unique_twice = [amt for amt, count in amount_counts.items() if count == 2]
    
    print(f"Amounts appearing 1x: {len(unique_once)}")
    print(f"Amounts appearing 2x: {len(unique_twice)}")
    print(f"Amounts appearing 2+ times: {len([amt for amt, count in amount_counts.items() if count >= 2])}")