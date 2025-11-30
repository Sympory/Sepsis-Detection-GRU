"""
Phase 3: Model Training for 56-Feature Model
============================================

Wrapper script for train_gru_v23.py that sets defaults
for the new 56-feature model training.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gru_v23 import main as original_main
from unittest.mock import patch

def main():
    print("="*60)
    print("PHASE 3: MODEL TRAINING (56 FEATURES)")
    print("="*60)
    
    # Set default arguments for Phase 3
    args = [
        'train_gru_v23.py',
        '--data', 'data/processed_56/',
        '--output', 'models/gru_v24_56features/',
        '--epochs', '60',
        '--batch-size', '512',
        '--lr', '0.001'
    ]
    
    # Patch sys.argv
    with patch.object(sys, 'argv', args):
        original_main()

if __name__ == '__main__':
    main()
