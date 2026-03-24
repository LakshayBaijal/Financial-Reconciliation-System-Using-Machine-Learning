import pandas as pd
from data_loader import DataLoader

print("=" * 80)
print("DIAGNOSTIC CHECK")
print("=" * 80)

# Load raw CSVs
print("\n[1] Loading raw CSV files...")
bank_raw = pd.read_csv('bank_statements.csv')
check_raw = pd.read_csv('check_register.csv')

print(f"Bank CSV shape: {bank_raw.shape}")
print(f"Check CSV shape: {check_raw.shape}")

print("\nBank CSV Columns:")
print(bank_raw.columns.tolist())
print("\nCheck CSV Columns:")
print(check_raw.columns.tolist())

print("\n" + "=" * 80)
print("[2] Bank Statement - First 3 rows:")
print("=" * 80)
print(bank_raw.head(3).to_string())

print("\n" + "=" * 80)
print("[3] Check Register - First 3 rows:")
print("=" * 80)
print(check_raw.head(3).to_string())

# Check data types
print("\n" + "=" * 80)
print("[4] Data Types:")
print("=" * 80)
print("\nBank data types:")
print(bank_raw.dtypes)
print("\nCheck data types:")
print(check_raw.dtypes)

# Clean and check
print("\n" + "=" * 80)
print("[5] After Cleaning:")
print("=" * 80)
loader = DataLoader()
bank_clean = loader.clean_data(bank_raw)
check_clean = loader.clean_data(check_raw)

print(f"Bank cleaned shape: {bank_clean.shape}")
print(f"Check cleaned shape: {check_clean.shape}")

print("\nBank amounts after cleaning:")
print(bank_clean['amount'].head(10).tolist())
print("\nCheck amounts after cleaning:")
print(check_clean['amount'].head(10).tolist())

# Find matching amounts
print("\n" + "=" * 80)
print("[6] Amount Matching Analysis:")
print("=" * 80)

bank_amounts = bank_clean['amount'].values
check_amounts = check_clean['amount'].values

print(f"Unique bank amounts: {len(set(bank_amounts))}")
print(f"Unique check amounts: {len(set(check_amounts))}")

# Find exact matches
exact_matches = set(bank_amounts) & set(check_amounts)
print(f"Exact amount matches: {len(exact_matches)}")
print(f"Sample exact matches: {sorted(list(exact_matches))[:20]}")

# Check with tolerance
print("\n" + "=" * 80)
print("[7] Amount Distribution:")
print("=" * 80)
print(f"Bank amount range: ${bank_clean['amount'].min():.2f} to ${bank_clean['amount'].max():.2f}")
print(f"Check amount range: ${check_clean['amount'].min():.2f} to ${check_clean['amount'].max():.2f}")

print("\nBank amount stats:")
print(bank_clean['amount'].describe())

print("\nCheck amount stats:")
print(check_clean['amount'].describe())