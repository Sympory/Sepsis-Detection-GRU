"""
FAST Synthetic Biomarker Data Generator (Vectorized) - REALISTIC NOISE VERSION
==============================================================================

Optimized version with INCREASED OVERLAP to prevent data leakage.
Generates 22 new clinical parameters for 1.5M rows.
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import os

def generate_fast_biomarkers(df):
    """Generate biomarkers using vectorized operations with realistic overlap"""
    n_samples = len(df)
    
    # Determine sepsis status
    if 'SepsisLabel' in df.columns:
        is_sepsis = df['SepsisLabel'].values == 1
    else:
        is_sepsis = np.random.random(n_samples) < 0.05
        
    # Determine severity (0.0 to 1.0)
    if 'Lactate' in df.columns:
        severity = df['Lactate'].fillna(1.0).values / 10.0
        severity = np.clip(severity, 0, 1)
    else:
        # Random severity distribution
        severity = np.random.beta(2, 5, n_samples)
        # Sepsis patients tend to have higher severity, but not always
        severity[is_sepsis] += np.random.beta(2, 2, np.sum(is_sepsis)) * 0.3
        severity = np.clip(severity, 0, 1)

    # Initialize result dictionary
    biomarkers = {}
    
    # Configuration with MORE OVERLAP
    # 'normal': (low, high)
    # 'sepsis': (low, high) -> Now overlaps significantly with normal
    configs = {
        'PCT': {'normal': (0.05, 0.5), 'sepsis': (0.3, 15.0), 'higher': True}, # Overlap 0.3-0.5
        'CRP': {'normal': (0, 20), 'sepsis': (10, 300), 'higher': True},       # Overlap 10-20
        'Presepsin': {'normal': (200, 800), 'sepsis': (500, 2000), 'higher': True}, # Overlap 500-800
        'IL6': {'normal': (0, 20), 'sepsis': (10, 500), 'higher': True},
        'IL1b': {'normal': (0, 10), 'sepsis': (5, 100), 'higher': True},
        'ESR': {'normal': (0, 30), 'sepsis': (15, 100), 'higher': True},
        'MDW': {'normal': (18, 24), 'sepsis': (20, 35), 'higher': True},
        'MPV': {'normal': (7.0, 12.0), 'sepsis': (7.5, 12.5), 'higher': False}, # Huge overlap
        'RDW': {'normal': (11.0, 15.5), 'sepsis': (13.0, 20.0), 'higher': True},
        'Neutrophils': {'normal': (1.5, 8.0), 'sepsis': (4.0, 30.0), 'higher': True},
        'Lymphocytes': {'normal': (0.8, 4.0), 'sepsis': (0.3, 2.5), 'higher': False},
        'DDimer': {'normal': (0, 1.0), 'sepsis': (0.5, 10.0), 'higher': True},
        'PT': {'normal': (10, 14.0), 'sepsis': (12, 25), 'higher': True},
        'aPTT': {'normal': (22, 38), 'sepsis': (30, 60), 'higher': True},
        'INR': {'normal': (0.8, 1.3), 'sepsis': (1.1, 3.0), 'higher': True},
        'IonizedCalcium': {'normal': (1.10, 1.35), 'sepsis': (0.8, 1.25), 'higher': False},
        'Phosphorus': {'normal': (2.5, 4.8), 'sepsis': (1.5, 5.5), 'higher': False}, # Messy
        'Albumin': {'normal': (3.2, 5.5), 'sepsis': (2.0, 4.0), 'higher': False},
        'Sodium': {'normal': (132, 148), 'sepsis': (125, 155), 'higher': False}, # Almost same
    }
    
    print("  Generating values with REALISTIC OVERLAP...")
    
    for name, cfg in configs.items():
        # 1. Generate baseline values (Normal distribution for everyone)
        # This ensures everyone starts from a "normal-ish" baseline
        n_low, n_high = cfg['normal']
        n_mean = (n_low + n_high) / 2
        n_std = (n_high - n_low) / 4
        
        values = np.random.normal(n_mean, n_std, n_samples)
        
        # 2. Shift values for Sepsis patients based on severity
        # But only shift SOME of them (not all sepsis patients have abnormal values)
        # 30% of sepsis patients might still have normal values for this specific marker
        
        s_low, s_high = cfg['sepsis']
        
        # Probability that this marker is abnormal in a sepsis patient
        # Some markers like PCT are very sensitive (0.9), others less so (0.6)
        sensitivity = 0.85 if name in ['PCT', 'CRP', 'Presepsin'] else 0.6
        
        # Mask for sepsis patients who actually show abnormal values
        abnormal_mask = is_sepsis & (np.random.random(n_samples) < sensitivity)
        
        if cfg['higher']:
            # Shift UP
            # Shift amount depends on severity
            # max_shift = difference between sepsis_high and normal_mean
            max_shift = s_high - n_mean
            shift = max_shift * severity * np.random.uniform(0.5, 1.5, n_samples)
            values[abnormal_mask] += shift[abnormal_mask]
        else:
            # Shift DOWN
            max_shift = n_mean - s_low
            shift = max_shift * severity * np.random.uniform(0.5, 1.5, n_samples)
            values[abnormal_mask] -= shift[abnormal_mask]
            
        # 3. Add random noise to everyone (measurement error, biological variation)
        noise_std = (n_high - n_low) * 0.2
        values += np.random.normal(0, noise_std, n_samples)
        
        # 4. Clip to physical limits
        values = np.maximum(values, 0.01)
        
        biomarkers[name] = values.round(2)

    # Derived markers
    print("  Calculating derived markers...")
    
    # NLR
    lym = np.maximum(biomarkers['Lymphocytes'], 0.1)
    biomarkers['NLR'] = (biomarkers['Neutrophils'] / lym).round(2)
    
    # PLR
    platelets = df['Platelets'].fillna(250).values if 'Platelets' in df.columns else np.full(n_samples, 250)
    biomarkers['PLR'] = (platelets / lym).round(2)
    
    # Anion Gap
    na = biomarkers['Sodium']
    cl = df['Chloride'].fillna(100).values if 'Chloride' in df.columns else np.full(n_samples, 100)
    hco3 = df['HCO3'].fillna(24).values if 'HCO3' in df.columns else np.full(n_samples, 24)
    biomarkers['AnionGap'] = (na - (cl + hco3)).round(2)
    
    return pd.DataFrame(biomarkers)

def main():
    parser = argparse.ArgumentParser(description='Fast Biomarker Generator')
    parser.add_argument('--input', default='data/train.csv')
    parser.add_argument('--output', default='data/train_56features.csv')
    args = parser.parse_args()
    
    print("="*60)
    print("FAST BIOMARKER GENERATOR (REALISTIC NOISE)")
    print("="*60)
    
    print(f"\nLoading data: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[OK] Loaded {len(df):,} rows")
    
    # Generate
    print("\nGenerating biomarkers...")
    start_time = datetime.now()
    
    biomarker_df = generate_fast_biomarkers(df)
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"[OK] Generated {len(biomarker_df.columns)} features in {duration:.2f} seconds")
    
    # Combine
    print("\nCombining datasets...")
    result_df = pd.concat([df, biomarker_df], axis=1)
    
    # Save
    print(f"Saving to: {args.output}")
    result_df.to_csv(args.output, index=False)
    
    print(f"\n[OK] COMPLETE! Total shape: {result_df.shape}")

if __name__ == '__main__':
    main()
