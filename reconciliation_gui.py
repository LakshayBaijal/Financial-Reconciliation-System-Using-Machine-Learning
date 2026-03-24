import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import threading
import traceback
import logging

# Import custom modules
from data_loader import DataLoader
from data_separator import DataSeparator
from feature_extractor import FeatureExtractor
from matcher import TransactionMatcher
from evaluator import Evaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReconciliationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Financial Reconciliation System')
        self.root.geometry('1000x800')
        self.root.configure(bg='#f0f0f0')
        
        self.bank_file = 'bank_statements.csv'
        self.check_file = 'check_register.csv'
        self.output_dir = Path('./output')
        self.output_dir.mkdir(exist_ok=True)
        
        self.loader = DataLoader()
        self.separator = DataSeparator()
        self.extractor = None
        self.evaluator = Evaluator()
        
        self.processing = False
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=15)
        
        title = tk.Label(title_frame, text='Financial Reconciliation System', 
                        font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title.pack()
        
        subtitle = tk.Label(title_frame, text='Match bank statements with check registers automatically',
                           font=('Arial', 10), bg='#f0f0f0', fg='gray')
        subtitle.pack()
        
        # File selection frame
        file_frame = tk.LabelFrame(self.root, text='Select Files', font=('Arial', 11, 'bold'),
                                   bg='#f0f0f0', padx=15, pady=15)
        file_frame.pack(fill='x', padx=20, pady=10)
        
        # Bank file
        bank_container = tk.Frame(file_frame, bg='#f0f0f0')
        bank_container.pack(fill='x', pady=5)
        tk.Label(bank_container, text='Bank Statements:', font=('Arial', 10), bg='#f0f0f0', width=20).pack(side='left')
        self.bank_label = tk.Label(bank_container, text='bank_statements.csv', fg='blue', bg='#f0f0f0', font=('Arial', 10))
        self.bank_label.pack(side='left', padx=10)
        tk.Button(bank_container, text='Browse', command=self.select_bank_file, width=10).pack(side='right')
        
        # Check file
        check_container = tk.Frame(file_frame, bg='#f0f0f0')
        check_container.pack(fill='x', pady=5)
        tk.Label(check_container, text='Check Register:', font=('Arial', 10), bg='#f0f0f0', width=20).pack(side='left')
        self.check_label = tk.Label(check_container, text='check_register.csv', fg='blue', bg='#f0f0f0', font=('Arial', 10))
        self.check_label.pack(side='left', padx=10)
        tk.Button(check_container, text='Browse', command=self.select_check_file, width=10).pack(side='right')
        
        # Settings frame
        settings_frame = tk.LabelFrame(self.root, text='Settings', font=('Arial', 11, 'bold'),
                                       bg='#f0f0f0', padx=15, pady=15)
        settings_frame.pack(fill='x', padx=20, pady=10)
        
        threshold_container = tk.Frame(settings_frame, bg='#f0f0f0')
        threshold_container.pack(fill='x')
        tk.Label(threshold_container, text='Confidence Threshold:', font=('Arial', 10), bg='#f0f0f0').pack(side='left')
        self.threshold_var = tk.IntVar(value=50)
        threshold_scale = tk.Scale(threshold_container, from_=0, to=100, orient='horizontal',
                                  variable=self.threshold_var, bg='#f0f0f0', length=300)
        threshold_scale.pack(side='left', padx=10)
        self.threshold_label = tk.Label(threshold_container, text='50%', bg='#f0f0f0', width=5, font=('Arial', 10))
        self.threshold_label.pack(side='left')
        threshold_scale.config(command=lambda v: self.threshold_label.config(text=f'{v}%'))
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=15)
        
        self.start_btn = tk.Button(button_frame, text='Start Reconciliation', command=self.start_reconciliation,
                                   bg='darkgreen', fg='white', font=('Arial', 11, 'bold'), padx=20, pady=8)
        self.start_btn.pack(side='left', padx=5)
        
        self.output_btn = tk.Button(button_frame, text='Open Output Folder', command=self.open_output,
                                    bg='blue', fg='white', font=('Arial', 10), padx=20, pady=8)
        self.output_btn.pack(side='left', padx=5)
        
        self.exit_btn = tk.Button(button_frame, text='Exit', command=self.root.quit,
                                  bg='darkred', fg='white', font=('Arial', 10), padx=20, pady=8)
        self.exit_btn.pack(side='left', padx=5)
        
        # Progress
        progress_frame = tk.Frame(self.root, bg='#f0f0f0')
        progress_frame.pack(pady=5)
        
        self.progress_var = tk.IntVar()
        self.progress_bar = tk.Label(progress_frame, text='Progress: 0%', bg='#f0f0f0', font=('Arial', 10))
        self.progress_bar.pack()
        
        # Output log
        log_label = tk.Label(self.root, text='Output Log:', font=('Arial', 10, 'bold'), bg='#f0f0f0')
        log_label.pack(anchor='w', padx=20, pady=(5, 0))
        
        self.output_text = scrolledtext.ScrolledText(self.root, height=15, width=120,
                                                     bg='black', fg='lightgreen', font=('Courier', 8))
        self.output_text.pack(padx=20, pady=10, fill='both', expand=True)
        self.output_text.config(state='disabled')
    
    def select_bank_file(self):
        """Select bank statements CSV"""
        file = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file:
            self.bank_file = file
            self.bank_label.config(text=os.path.basename(file), fg='green')
    
    def select_check_file(self):
        """Select check register CSV"""
        file = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if file:
            self.check_file = file
            self.check_label.config(text=os.path.basename(file), fg='green')
    
    def log_output(self, message: str):
        """Log message to output window (thread-safe)"""
        self.output_text.config(state='normal')
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.output_text.insert('end', f'[{timestamp}] {message}\n')
        self.output_text.see('end')
        self.output_text.config(state='disabled')
        self.root.update()
        logger.info(message)
    
    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_var.set(value)
        self.progress_bar.config(text=f'Progress: {value}%')
        self.root.update()
    
    def start_reconciliation(self):
        """Start reconciliation in separate thread"""
        if self.processing:
            messagebox.showwarning('Warning', 'Reconciliation already in progress!')
            return
        
        if not self.bank_file or not self.check_file:
            messagebox.showerror('Error', 'Please select both CSV files!')
            return
        
        self.start_btn.config(state='disabled')
        self.processing = True
        
        # Run in separate thread
        thread = threading.Thread(target=self.process_reconciliation)
        thread.daemon = True
        thread.start()
    
    def process_reconciliation(self):
        """Main reconciliation process"""
        try:
            threshold = self.threshold_var.get() / 100.0
            
            self.log_output('[STEP 1] Loading bank statements...')
            bank_df = self.loader.load_csv(self.bank_file)
            self.log_output(f'[OK] Loaded {len(bank_df)} bank transactions')
            
            self.log_output('[STEP 2] Loading check register...')
            check_df = self.loader.load_csv(self.check_file)
            self.log_output(f'[OK] Loaded {len(check_df)} check transactions')
            
            self.update_progress(10)
            
            # Validate
            if not self.loader.validate_bank_data(bank_df):
                self.log_output('[ERROR] Bank file missing required columns: date, description, amount, type')
                return
            if not self.loader.validate_check_data(check_df):
                self.log_output('[ERROR] Check file missing required columns')
                return
            
            # Clean
            self.log_output('[STEP 3] Cleaning data...')
            bank_df = self.loader.clean_data(bank_df)
            check_df = self.loader.clean_data(check_df)
            self.log_output(f'[OK] Data cleaned: {len(bank_df)} bank, {len(check_df)} check')
            
            self.update_progress(20)
            
            # Separate data
            self.log_output('[STEP 4] Separating training and testing data...')
            training_pairs, testing_candidates = self.separator.separate_by_amount_uniqueness(
                bank_df, check_df
            )
            self.log_output(f'[OK] Training pairs: {len(training_pairs)} | Testing: {len(testing_candidates)}')
            
            self.update_progress(30)
            
            # Initialize extractor
            self.log_output('[STEP 5] Initializing embedding model (downloading ~120MB on first run)...')
            self.log_output('[...] Please wait, model is downloading and caching...')
            self.extractor = FeatureExtractor(model_name='all-MiniLM-L6-v2')
            self.log_output('[OK] Embedding model ready')
            
            self.update_progress(50)
            
            # Match transactions
            self.log_output('[STEP 6] Matching transactions using Hungarian algorithm...')
            matcher = TransactionMatcher(training_pairs, self.extractor)
            matched_results = matcher.match_all_transactions(bank_df, check_df, threshold)
            self.log_output(f'[OK] Matching complete')
            
            self.update_progress(70)
            
            # Calculate metrics
            self.log_output('[STEP 7] Calculating metrics...')
            metrics = self.evaluator.calculate_metrics(matched_results)
            report = self.evaluator.generate_report(metrics)
            self.log_output('[OK] Metrics calculated')
            
            self.update_progress(85)
            
            # Save results
            self.log_output('[STEP 8] Saving results...')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.output_dir / f'matched_transactions_{timestamp}.csv'
            matched_results.to_csv(results_file, index=False)
            self.log_output(f'[OK] Results saved: {results_file.name}')
            
            report_file = self.output_dir / f'reconciliation_report_{timestamp}.txt'
            with open(report_file, 'w') as f:
                f.write(report)
            self.log_output(f'[OK] Report saved: {report_file.name}')
            
            self.update_progress(100)
            
            # Display summary
            self.log_output('=' * 80)
            self.log_output('[SUCCESS] RECONCILIATION COMPLETE!')
            self.log_output('=' * 80)
            self.log_output(f'\n[SUMMARY]')
            self.log_output(f'  Total Transactions: {metrics["total_transactions"]}')
            self.log_output(f'  Matched: {metrics["matched_count"]}')
            self.log_output(f'  Unmatched: {metrics["unmatched_count"]}')
            self.log_output(f'  Match Rate: {metrics["match_rate"]:.1f}%')
            self.log_output(f'  Avg Confidence: {metrics["avg_confidence_score"]:.4f}')
            self.log_output(f'  Precision: {metrics["precision"]:.4f}')
            self.log_output(f'  Recall: {metrics["recall"]:.4f}')
            self.log_output(f'  F1 Score: {metrics["f1_score"]:.4f}')
            self.log_output(f'\n[OUTPUT] Files saved to: {self.output_dir.absolute()}')
            
            messagebox.showinfo('Success', f'Reconciliation completed!\n\nMatched: {metrics["matched_count"]}/{metrics["total_transactions"]}\nMatch Rate: {metrics["match_rate"]:.1f}%')
            
        except Exception as e:
            self.log_output(f'[ERROR] {str(e)}')
            self.log_output(f'[ERROR] {traceback.format_exc()}')
            messagebox.showerror('Error', f'Reconciliation failed:\n{str(e)}\n\nCheck the log for details.')
        
        finally:
            self.processing = False
            self.start_btn.config(state='normal')
    
    def open_output(self):
        """Open output folder"""
        if self.output_dir.exists():
            os.startfile(str(self.output_dir.absolute()))
        else:
            messagebox.showinfo('Info', 'Output folder does not exist yet.')

def main():
    root = tk.Tk()
    app = ReconciliationGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()