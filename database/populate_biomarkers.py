"""
Populate Existing Data with Synthetic Biomarker Values
=======================================================

Generates realistic biomarker values for existing patients based on:
- Normal ranges for healthy patients
- Elevated values for sepsis-positive patients
- Clinical correlations between biomarkers
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from auth import DB_CONFIG


def generate_sepsis_biomarkers(is_sepsis=False, severity='mild'):
    """
    Generate realistic biomarker values based on sepsis status
    
    Args:
        is_sepsis: Boolean, whether patient has sepsis
        severity: 'mild', 'moderate', 'severe' for sepsis patients
    
    Returns:
        dict with biomarker values
    """
    
    if not is_sepsis:
        # Normal ranges
        biomarkers = {
            'sepsis_markers': {
                'PCT': np.random.uniform(0.01, 0.3),     # Normal <0.5
                'CRP': np.random.uniform(1, 8),          # Normal <10
                'Presepsin': np.random.uniform(100, 400), # Normal <600
                'IL6': np.random.uniform(0.5, 5),        # Normal <7
                'IL1b': np.random.uniform(0.5, 3)        # Normal <5
            },
            'hematology': {
                'ESR': np.random.uniform(2, 15),         # Normal <20
                'MDW': np.random.uniform(19, 21),        # Normal 19-21
                'MPV': np.random.uniform(7.5, 11),       # Normal 7.5-11.5
                'RDW': np.random.uniform(11.5, 14)       # Normal 11.5-14.5
            },
            'coagulation': {
                'DDimer': np.random.uniform(0.1, 0.4),   # Normal <0.5
                'Fibrinogen': np.random.uniform(220, 380), # Normal 200-400
                'PT': np.random.uniform(11, 13),         # Normal 11-13.5
                'aPTT': np.random.uniform(25, 35),      # Normal 25-35
                'INR': np.random.uniform(0.9, 1.1)      # Normal 0.8-1.2
            },
            'chemistry': {
                'IonizedCalcium': np.random.uniform(1.18, 1.30), # Normal 1.16-1.32
                'Magnesium': np.random.uniform(1.8, 2.3),        # Normal 1.7-2.4
                'Phosphorus': np.random.uniform(2.5, 4.5),       # Normal 2.5-4.5
                'Albumin': np.random.uniform(3.8, 5.0)           # Normal 3.5-5.5
            }
        }
    else:
        # Sepsis ranges - vary by severity
        if severity == 'mild':
            multiplier = 1.5
        elif severity == 'moderate':
            multiplier = 3.0
        else:  # severe
            multiplier = 5.0
        
        biomarkers = {
            'sepsis_markers': {
                'PCT': np.random.uniform(2, 10) * multiplier,      # Sepsis >2
                'CRP': np.random.uniform(50, 150) * (multiplier/2), # Sepsis >50
                'Presepsin': np.random.uniform(800, 2000),         # Sepsis >600
                'IL6': np.random.uniform(20, 100) * (multiplier/2), # Elevated
                'IL1b': np.random.uniform(10, 50)                  # Elevated
            },
            'hematology': {
                'ESR': np.random.uniform(30, 100),                 # Elevated
                'MDW': np.random.uniform(23, 28),                  # Sepsis >23
                'MPV': np.random.uniform(8, 13),                   # Variable
                'RDW': np.random.uniform(15, 20)                   # Elevated
            },
            'coagulation': {
                'DDimer': np.random.uniform(1.5, 8.0) * (multiplier/3), # DIC
                'Fibrinogen': np.random.uniform(100, 180) if severity == 'severe' else np.random.uniform(400, 600),
                'PT': np.random.uniform(14, 20),                   # Prolonged
                'aPTT': np.random.uniform(38, 60),                # Prolonged
                'INR': np.random.uniform(1.3, 2.5)                # Elevated
            },
            'chemistry': {
                'IonizedCalcium': np.random.uniform(0.9, 1.10),    # Hypocalcemia
                'Magnesium': np.random.uniform(1.2, 1.6),          # Hypomagnesemia
                'Phosphorus': np.random.uniform(1.5, 2.3),         # Hypophosphatemia
                'Albumin': np.random.uniform(2.0, 3.2)             # Hypoalbuminemia
            }
        }
    
    return biomarkers


def calculate_derived_values(vital_signs, biomarkers):
    """Calculate NLR, PLR, Anion Gap, Shock Index, etc."""
    
    derived = {}
    
    # Extract values from vital signs (if they exist)
    hr = vital_signs.get('HR')
    sbp = vital_signs.get('SBP')
    neutrophils = vital_signs.get('Neutrophils')  # Assume exists or generate
    lymphocytes = vital_signs.get('Lymphocytes', np.random.uniform(1.0, 3.0))
    platelets = vital_signs.get('Platelets', np.random.uniform(150, 400))
    sodium = vital_signs.get('Na', np.random.uniform(135, 145))
    chloride = vital_signs.get('Cl', np.random.uniform(98, 106))
    bicarbonate = vital_signs.get('HCO3', np.random.uniform(22, 28))
    
    # Calculate NLR (Neutrophil-Lymphocyte Ratio)
    if neutrophils and lymphocytes and lymphocytes > 0:
        derived['NLR'] = round(neutrophils / lymphocytes, 2)
    else:
        # If not available, generate based on typical values
        derived['NLR'] = round(np.random.uniform(2, 8), 2)
    
    # Calculate PLR (Platelet-Lymphocyte Ratio)
    if platelets and lymphocytes and lymphocytes > 0:
        derived['PLR'] = round(platelets / lymphocytes, 2)
    else:
        derived['PLR'] = round(np.random.uniform(100, 250), 2)
    
    # Calculate Anion Gap
    if sodium and chloride and bicarbonate:
        derived['AnionGap'] = round(sodium - (chloride + bicarbonate), 2)
    else:
        derived['AnionGap'] = round(np.random.uniform(8, 16), 2)
    
    # Calculate Shock Index
    if hr and sbp and sbp > 0:
        derived['ShockIndex'] = round(hr / sbp, 2)
    
    # Calculate simplified SOFA score (placeholder - needs more inputs)
    derived['SOFA'] = int(np.random.randint(0, 8))
    derived['qSOFA'] = int(np.random.randint(0, 3))
    
    return derived


def populate_biomarkers():
    """Populate existing hourly_data with biomarker values"""
    
    print("=" * 60)
    print("POPULATING BIOMARKER DATA")
    print("=" * 60)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get all hourly data entries
        cur.execute("""
            SELECT h.id, h.patient_id, h.vital_signs, h.prediction, h.risk_level
            FROM hourly_data h
            ORDER BY h.patient_id, h.hour
        """)
        
        entries = cur.fetchall()
        print(f"\n[1/2] Found {len(entries)} hourly data entries")
        
        updated_count = 0
        
        for entry in entries:
            entry_id = entry['id']
            vital_signs = json.loads(entry['vital_signs']) if isinstance(entry['vital_signs'], str) else entry['vital_signs']
            prediction = entry['prediction']
            
            # Determine sepsis status and severity
            is_sepsis = prediction is not None and prediction > 0.5
            
            if prediction is None:
                severity = 'mild'
            elif prediction > 0.7:
                severity = 'severe'
            elif prediction > 0.5:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            # Generate biomarkers
            biomarkers = generate_sepsis_biomarkers(is_sepsis, severity)
            
            # Calculate derived values
            derived = calculate_derived_values(vital_signs, biomarkers)
            
            # Add derived values to biomarkers
            biomarkers['hematology'].update({
                'NLR': derived.get('NLR'),
                'PLR': derived.get('PLR')
            })
            biomarkers['chemistry']['AnionGap'] = derived.get('AnionGap')
            biomarkers['scores'] = {
                'ShockIndex': derived.get('ShockIndex'),
                'SOFA': derived.get('SOFA'),
                'qSOFA': derived.get('qSOFA')
            }
            
            # Update database
            cur.execute("""
                UPDATE hourly_data 
                SET vital_signs_extended = %s
                WHERE id = %s
            """, (json.dumps(biomarkers), entry_id))
            
            updated_count += 1
            
            if updated_count % 100 == 0:
                print(f"  Processed {updated_count}/{len(entries)} entries...")
        
        conn.commit()
        
        print(f"\n[2/2] Successfully updated {updated_count} entries")
        print("\n" + "=" * 60)
        print("✓ BIOMARKER DATA POPULATED!")
        print("=" * 60)
        
        # Show sample
        cur.execute("""
            SELECT vital_signs_extended 
            FROM hourly_data 
            WHERE vital_signs_extended IS NOT NULL 
            LIMIT 1
        """)
        sample = cur.fetchone()
        
        if sample:
            print("\n📊 Sample biomarker data:")
            print(json.dumps(sample['vital_signs_extended'], indent=2))
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
        
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    success = populate_biomarkers()
    sys.exit(0 if success else 1)
