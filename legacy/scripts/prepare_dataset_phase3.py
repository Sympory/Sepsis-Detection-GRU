"""
Phase 3: Data Preparation for 56-Feature Model
==============================================

Wrapper script for prepare_sequence_dataset_v23.py that sets defaults
for the new 56-feature dataset.
"""

import os
import sys

# Add parent directory to path to import existing scripts if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_sequence_dataset_v23 import main as original_main
from unittest.mock import patch

def main():
    print("="*60)
    print("PHASE 3: DATASET PREPARATION (56 FEATURES)")
    print("="*60)
    
    # Set default arguments for Phase 3
    args = [
        'prepare_sequence_dataset_v23.py',
        '--input', 'data/train_56features.csv',
        '--output', 'data/processed_56/',
        '--window', '6',
        '--step', '1'
    ]
    
    # Patch sys.argv
    with patch.object(sys, 'argv', args):
        original_main()

if __name__ == '__main__':
    main()
