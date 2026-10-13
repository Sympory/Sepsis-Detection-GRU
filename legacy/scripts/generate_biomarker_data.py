"""
Synthetic Biomarker Data Generator for Sepsis Prediction
=========================================================

This script generates realistic synthetic biomarker values for the existing
train.csv dataset, adding 22 new clinical parameters based on:
- Medical reference ranges
- Sepsis status correlation
- Inter-biomarker correlations

New Features (22):
- Sepsis Markers: PCT, CRP, Presepsin, IL6, IL1b
- Hematology: ESR, MDW, MPV, RDW, Neutrophils, Lymphocytes  
- Coagulation: DDimer, PT, aPTT, INR
- Chemistry: IonizedCalcium, Phosphorus, Albumin, Sodium
- Calculated: NLR, PLR, Anion Gap

Usage:
    python generate_biomarker_data.py --input data/train.csv --output data/train_56features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)


class BiomarkerGenerator:
    """Generate synthetic biomarker values based on sepsis status"""
    
    def __init__(self):
        # Define reference ranges and sepsis correlation
        self.biomarker_config = {
            # Sepsis Markers
            'PCT': {
                'normal': (0.05, 0.3), 'sepsis': (2.0, 15.0), 
                'unit': 'ng/mL', 'higher_in_sepsis': True
            },
            'CRP': {
                'normal': (0, 10), 'sepsis': (50, 300),
                'unit': 'mg/L', 'higher_in_sepsis': True
            },
            'Presepsin': {
                'normal': (200, 600), 'sepsis': (800, 2000),
                'unit': 'pg/mL', 'higher_in_sepsis': True
            },
            'IL6': {
                'normal': (0, 7), 'sepsis': (50, 500),
                'unit': 'pg/mL', 'higher_in_sepsis': True
            },
            'IL1b': {
                'normal': (0, 5), 'sepsis': (20, 100),
                'unit': 'pg/mL', 'higher_in_sepsis': True
            },
            
            # Hematology Extended
            'ESR': {
                'normal': (0, 20), 'sepsis': (30, 100),
                'unit': 'mm/hr', 'higher_in_sepsis': True
            },
            'MDW': {
                'normal': (19, 21), 'sepsis': (23, 35),
                'unit': 'unit', 'higher_in_sepsis': True
            },
            'MPV': {
                'normal': (7.5, 11.5), 'sepsis': (8, 12),
                'unit': 'fL', 'higher_in_sepsis': False
            },
            'RDW': {
                'normal': (11.5, 14.5), 'sepsis': (14, 20),
                'unit': '%', 'higher_in_sepsis': True
            },
            'Neutrophils': {
                'normal': (2, 7), 'sepsis': (10, 30),
                'unit': 'K/µL', 'higher_in_sepsis': True
            },
            'Lymphocytes': {
                'normal': (1, 3), 'sepsis': (0.3, 1.5),
                'unit': 'K/µL', 'higher_in_sepsis': False
            },
            
            # Coagulation Extended
            'DDimer': {
                'normal': (0, 0.5), 'sepsis': (1.5, 10),
                'unit': 'μg/mL', 'higher_in_sepsis': True
            },
            'PT': {
                'normal': (11, 13.5), 'sepsis': (14, 25),
                'unit': 'seconds', 'higher_in_sepsis': True
            },
            'aPTT': {
                'normal': (25, 35), 'sepsis': (35, 60),
                'unit': 'seconds', 'higher_in_sepsis': True
            },
            'INR': {
                'normal': (0.8, 1.2), 'sepsis': (1.3, 3.0),
                'unit': 'ratio', 'higher_in_sepsis': True
            },
            
            # Chemistry Extended
            'IonizedCalcium': {
                'normal': (1.16, 1.32), 'sepsis': (0.8, 1.1),
                'unit': 'mmol/L', 'higher_in_sepsis': False
            },
            'Phosphorus': {
                'normal': (2.5, 4.5), 'sepsis': (1.5, 6.0),
                'unit': 'mg/dL', 'higher_in_sepsis': False
            },
            'Albumin': {
                'normal': (3.5, 5.5), 'sepsis': (2.0, 3.2),
                'unit': 'g/dL', 'higher_in_sepsis': False
            },
            'Sodium': {
                'normal': (135, 145), 'sepsis': (125, 155),
                'unit': 'mmol/L', 'higher_in_sepsis': False
            },
        }
    
    def generate_value(self, biomarker, is_sepsis, severity=None):
        """
        Generate a single biomarker value
        
        Args:
            biomarker: Biomarker name
            is_sepsis: Boolean, whether patient has sepsis
            severity: Optional, sepsis severity (0-1)
        
        Returns:
            Generated value
        """
        config = self.biomarker_config[biomarker]
        
        if is_sepsis:
            # Use sepsis range
            low, high = config['sepsis']
            # If severity provided, adjust range
            if severity is not None:
                # Higher severity → more extreme values
                mid = (low + high) / 2
                if config['higher_in_sepsis']:
                    low = mid + (high - mid) * (severity * 0.5)
                else:
                    high = mid - (mid - low) * (severity * 0.5)
        else:
            # Use normal range
            low, high = config['normal']
        
        # Generate value with some noise
        mean = (low + high) / 2
        std = (high - low) / 4  # 95% within range
        
        # Truncated normal distribution
        value = stats.truncnorm(
            (low - mean) / std,
            (high - mean) / std,
            loc=mean,
            scale=std
        ).rvs()
        
        return round(value, 2)
    
    def calculate_derived_markers(self, neutrophils, lymphocytes, platelets, 
                                   sodium, chloride, hco3):
        """
        Calculate derived biomarkers
        
        Returns:
            dict with NLR, PLR, AnionGap
        """
        nlr = neutrophils / max(lymphocytes, 0.1) if lymphocytes > 0 else 10
        plr = platelets / max(lymphocytes, 0.1) if lymphocytes > 0 else 200
        anion_gap = sodium - (chloride + hco3)
        
        return {
            'NLR': round(nlr, 2),
            'PLR': round(plr, 2),
            'AnionGap': round(anion_gap, 2)
        }
    
    def generate_row_biomarkers(self, row):
        """Generate all biomarkers for a single row"""
        
        # Determine sepsis status
        # Assume 'SepsisLabel' column exists in train.csv
        is_sepsis = row.get('SepsisLabel', 0) == 1
        
        # Estimate severity from existing features (if available)
        # Use Lactate as severity proxy
        severity = None
        if 'Lactate' in row and pd.notna(row['Lactate']):
            severity = min(row['Lactate'] / 10.0, 1.0)  # Normalize 0-1
        
        # Generate each biomarker
        biomarkers = {}
        for biomarker in self.biomarker_config.keys():
            biomarkers[biomarker] = self.generate_value(biomarker, is_sepsis, severity)
        
        # Calculate derived markers (need existing features)
        neutrophils = biomarkers['Neutrophils']
        lymphocytes = biomarkers['Lymphocytes']
        platelets = row.get('Platelets', 200)  # Default if missing
        sodium = biomarkers['Sodium']
        chloride = row.get('Chloride', 100)  # From existing data
        hco3 = row.get('HCO3', 24)  # From existing data
        
        derived = self.calculate_derived_markers(
            neutrophils, lymphocytes, platelets,
            sodium, chloride, hco3
        )
        biomarkers.update(derived)
        
        return biomarkers


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic biomarker data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/train.csv',
        help='Input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/train_56features.csv',
        help='Output CSV file with biomarkers'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Process only N rows (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SYNTHETIC BIOMARKER GENERATOR")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print(f"\nLoading data: {args.input}")
    df = pd.read_csv(args.input)
    
    if args.sample:
        print(f"Sampling {args.sample} rows for testing...")
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
    
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"Existing columns: {df.columns.tolist()[:10]}...")
    
    # Initialize generator
    generator = BiomarkerGenerator()
    print(f"\nGenerating {len(generator.biomarker_config) + 3} new biomarkers...")
    
    # Generate biomarkers
    biomarker_data = []
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx:,}/{len(df):,} rows...")
        
        biomarkers = generator.generate_row_biomarkers(row)
        biomarker_data.append(biomarkers)
    
    # Create biomarker dataframe
    biomarker_df = pd.DataFrame(biomarker_data)
    
    # Combine with original data
    result_df = pd.concat([df, biomarker_df], axis=1)
    
    print(f"\n[OK] Generated biomarkers")
    print(f"  New columns: {list(biomarker_df.columns)}")
    print(f"  Total columns: {len(result_df.columns)} (was {len(df.columns)})")
    
    # Save
    print(f"\nSaving to: {args.output}")
    result_df.to_csv(args.output, index=False)
    
    print(f"\n[OK] COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {args.output}")
    print(f"Shape: {result_df.shape}")


if __name__ == '__main__':
    main()
