# Financial Reconciliation System

## Demo Video

https://github.com/user-attachments/assets/0d097a49-fd48-4c38-9754-755405cde93b

## Report Link
```br
https://github.com/LakshayBaijal/Financial-Reconciliation-System-Using-Machine-Learning/blob/main/REPORT.md
```


An AI-powered system that automatically matches transactions between bank statements and check registers using BERT embeddings and the Hungarian algorithm.

## Quick Start

```bash
pip install -r requirements.txt
python reconcile_quick.py          # CLI mode (recommended)
python reconciliation_gui.py        # GUI mode
```

## Overview

**Problem**: Match 308 bank transactions to 308 check registers despite different descriptions, dates, and amounts.

**Solution**:
- Phase 1: Auto-match transactions with unique amounts (286 pairs, 100% confidence)
- Phase 2: ML-match remaining transactions using BERT embeddings + Hungarian algorithm (22 candidates)

**Results**:
- Match Rate: 93.51% (288/308)
- Precision: 1.0000 (100% accurate)
- Recall: 0.9351 (93.51% found)
- F1 Score: 0.9664 (96.64%)

<img width="726" height="357" alt="Screenshot 2026-03-25 042212" src="https://github.com/user-attachments/assets/6ce627d4-3d94-4b8b-a7b0-05cb8b9af1bd" />



## Project Structure

```
├── README.md                      # Quick start (this file)
├── REPORT.md                      # Detailed analysis & assignment requirements
├── requirements.txt               # Dependencies
├── bank_statements.csv            # Input: 308 bank transactions
├── check_register.csv             # Input: 308 check register transactions
│
├── data_loader.py                 # CSV loading & validation
├── data_separator.py              # Train/test separation (unique vs non-unique amounts)
├── feature_extractor.py           # BERT embeddings (all-MiniLM-L6-v2)
├── matcher.py                     # Transaction matching (Hungarian algorithm)
├── evaluator.py                   # Metrics calculation (Precision/Recall/F1)
│
├── reconcile_quick.py             # CLI runner
├── reconciliation_gui.py           # GUI runner
│
└── output/                        # Results saved here
    ├── matched_transactions_*.csv
    └── reconciliation_report_*.txt
```

## How It Works

### Phase 1: Automatic Matching
- Identify amounts appearing exactly once in both datasets
- These MUST match (100% confidence)
- Result: 286 guaranteed correct training pairs

### Phase 2: ML-Based Matching
1. Generate BERT embeddings for all descriptions (384-dim vectors)
2. Build 308×308 similarity matrix (text + amount similarity)
3. Apply Hungarian algorithm for optimal 1-to-1 assignment
4. Filter by confidence threshold (0.5)
5. Result: 2 additional matches from 12 challenging cases, 20 unmatched

## Implementation

### Core Modules

**data_loader.py**: Load CSV, validate columns, clean data

**data_separator.py**: Split into Phase 1 (286 unique) and Phase 2 (22 non-unique)

**feature_extractor.py**: Generate BERT embeddings (all-MiniLM-L6-v2)
- Model: 384-dimensional pre-trained embeddings
- Pre-trained on 1B sentence pairs
- ~1000 texts/second on CPU

**matcher.py**: Match transactions
- Build 308×308 similarity matrix
- Hungarian algorithm (O(n³), ~1 sec)
- Filter by confidence (default 0.5)

**evaluator.py**: Calculate metrics
- Ground truth: Transaction ID alignment (B0047 ↔ R0047)
- TP=288, FP=0, FN=20
- Precision=1.0, Recall=0.9351, F1=0.9664

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate              # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running

### CLI Mode
```bash
python reconcile_quick.py
```

Output:
- `matched_transactions_*.csv` - Matched transactions with confidence
- `reconciliation_report_*.txt` - Detailed metrics and analysis

### GUI Mode
```bash
python reconciliation_gui.py
```

Features: Confidence slider, real-time log, success dialog

## Performance Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| Match Rate | 93.51% | Matched 288/308 transactions |
| Precision | 1.0000 | 100% of matches correct |
| Recall | 0.9351 | Found 93.51% of possible matches |
| F1 Score | 0.9664 | 96.64% overall performance |

**Confidence Distribution**:
- High (≥0.85): 4 transactions (1.39%)
- Medium (0.70-0.85): 71 transactions (24.65%)
- Low (0.50-0.70): 213 transactions (73.96%)
- Average: 0.6485

## Design Decisions

### Why BERT Instead of SVD (Paper)?
- Better semantic understanding (neural vs term-frequency)
- Pre-trained on 1B sentence pairs (general knowledge)
- 93.51% accuracy vs ~85% with SVD
- More maintainable (no manual curation)

### Why Hungarian Algorithm?
- Globally optimal 1-to-1 matching (not greedy)
- Prevents cascading errors
- O(n³) acceptable for 308 transactions

### Why Two-Phase Approach?
- Phase 1: Perfect ground truth on unique amounts (286 pairs)
- Phase 2: Handles difficult non-unique cases (22)
- Separates easy from hard problems

### Why Transaction ID as Ground Truth?
- Both datasets synthetically generated
- IDs clearly encode matches (B0047 ↔ R0047)
- Unambiguous, verifiable validation

## Limitations

1. **20 Unmatched Transactions**: Below confidence threshold, difficult cases
2. **Low Avg Confidence (0.6485)**: Descriptions vary significantly, system appropriately conservative
3. **No Date Proximity**: Only uses text + amount (could add date features)
4. **General Model**: Not trained on financial terminology (could use FinBERT)
5. **1-to-1 Only**: Doesn't handle split/bundled transactions
6. **No User Loop**: Doesn't learn from corrections

## Future Improvements

**High Priority**:
- Add date proximity scoring (+1-2% accuracy)
- Financial domain embeddings (+2-3% accuracy)
- Batch testing on real data (validation)

**Medium Priority**:
- Interactive web GUI (UX)
- Active learning loop (continuous improvement)
- Advanced matching strategies (complex cases)

**Low Priority**:
- Performance optimization (10-50x speedup)
- Explainability features (why matches made)
- Edge case handling (duplicates, reversals)

## Metrics Explained

**Precision** = TP/(TP+FP) = 288/288 = 1.0000
- Of all matches made, what % are correct?
- Our result: 100% of matches are correct

**Recall** = TP/(TP+FN) = 288/308 = 0.9351
- Of all possible matches, what % did we find?
- Our result: Found 93.51%, missed 20 (6.49%)

**F1 Score** = 2×(P×R)/(P+R) = 0.9664
- Harmonic mean balancing precision and recall
- Our result: 96.64% overall performance

## Ground Truth

We use transaction ID alignment as ground truth:
- Bank: B0047, Check: R0047
- Numeric portion: 0047 == 0047 → CORRECT
- This provides unambiguous, verifiable validation

## Time to Execute

- First run: ~5-8 seconds (+ 2 min BERT download)
- Subsequent runs: ~5-8 seconds

## Technical Stack

- **Language**: Python 3.8+
- **ML**: Sentence Transformers (BERT all-MiniLM-L6-v2)
- **Algorithm**: Hungarian algorithm (optimal assignment)
- **Libraries**: Pandas, NumPy, scikit-learn, SciPy

---

For detailed analysis covering:
- Performance breakdown (which types hardest to match)
- Design decisions & trade-offs
- Limitations & future improvements
- Implementation details of each component

See `REPORT.md` (assignment requirement documentation)
