-- ============================================================================
-- DATABASE SCHEMA EXTENSION - Phase 2: Clinical Biomarkers
-- ============================================================================
-- Adds 22 new clinical biomarkers for enhanced sepsis detection
-- ============================================================================

-- Backup existing table (safety first!)
-- Run this before making changes
DROP TABLE IF EXISTS hourly_data_backup;
CREATE TABLE hourly_data_backup AS SELECT * FROM hourly_data;

-- Add new biomarker columns to hourly_data table
ALTER TABLE hourly_data ADD COLUMN IF NOT EXISTS vital_signs_extended JSONB DEFAULT '{}';

-- Update vital_signs_extended to include new biomarkers
COMMENT ON COLUMN hourly_data.vital_signs_extended IS 
'Extended biomarker data in JSON format:
{
  "sepsis_markers": {
    "PCT": float,       // Procalcitonin (ng/mL)
    "CRP": float,       // C-Reactive Protein (mg/L)
    "Presepsin": float, // sCD14-ST (pg/mL)
    "IL6": float,       // Interleukin-6 (pg/mL)
    "IL1b": float       // Interleukin-1β (pg/mL)
  },
  "hematology": {
    "ESR": float,       // Erythrocyte Sedimentation Rate (mm/hr)
    "MDW": float,       // Monocyte Distribution Width
    "NLR": float,       // Neutrophil-Lymphocyte Ratio (calculated)
    "PLR": float,       // Platelet-Lymphocyte Ratio (calculated)
    "MPV": float,       // Mean Platelet Volume (fL)
    "RDW": float        // Red Cell Distribution Width (%)
  },
  "coagulation": {
    "DDimer": float,    // D-Dimer (μg/mL)
    "Fibrinogen": float,// Fibrinogen (mg/dL)
    "PT": float,        // Prothrombin Time (seconds)
    "aPTT": float,      // activated Partial Thromboplastin Time
    "INR": float        // International Normalized Ratio
  },
  "chemistry": {
    "AnionGap": float,  // Anion Gap (calculated)
    "IonizedCalcium": float, // mmol/L
    "Magnesium": float, // mg/dL
    "Phosphorus": float,// mg/dL
    "Albumin": float    // g/dL
  },
  "scores": {
    "SOFA": int,        // Sequential Organ Failure Assessment
    "qSOFA": int,       // Quick SOFA
    "ShockIndex": float // HR/SBP (calculated)
  }
}';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_hourly_vital_extended ON hourly_data USING GIN (vital_signs_extended);

-- ============================================================================
-- CALCULATED FIELDS - Automatic computation functions
-- ============================================================================

-- Function to calculate NLR (Neutrophil-Lymphocyte Ratio)
CREATE OR REPLACE FUNCTION calculate_nlr(neutrophils FLOAT, lymphocytes FLOAT)
RETURNS FLOAT AS $$
BEGIN
    IF lymphocytes IS NULL OR lymphocytes = 0 THEN
        RETURN NULL;
    END IF;
    RETURN neutrophils / lymphocytes;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate PLR (Platelet-Lymphocyte Ratio)
CREATE OR REPLACE FUNCTION calculate_plr(platelets FLOAT, lymphocytes FLOAT)
RETURNS FLOAT AS $$
BEGIN
    IF lymphocytes IS NULL OR lymphocytes = 0 THEN
        RETURN NULL;
    END IF;
    RETURN platelets / lymphocytes;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate Anion Gap
CREATE OR REPLACE FUNCTION calculate_anion_gap(sodium FLOAT, chloride FLOAT, bicarbonate FLOAT)
RETURNS FLOAT AS $$
BEGIN
    IF sodium IS NULL OR chloride IS NULL OR bicarbonate IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN sodium - (chloride + bicarbonate);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate Shock Index
CREATE OR REPLACE FUNCTION calculate_shock_index(hr FLOAT, sbp FLOAT)
RETURNS FLOAT AS $$
BEGIN
    IF sbp IS NULL OR sbp = 0 THEN
        RETURN NULL;
    END IF;
    RETURN hr / sbp;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate SOFA score (simplified version)
CREATE OR REPLACE FUNCTION calculate_sofa_score(
    pao2 FLOAT,
    fio2 FLOAT,
    platelets FLOAT,
    bilirubin FLOAT,
    map FLOAT,
    gcs INT,
    creatinine FLOAT,
    urine_output FLOAT
) RETURNS INT AS $$
DECLARE
    sofa_score INT := 0;
    pao2_fio2 FLOAT;
BEGIN
    -- Respiration (PaO2/FiO2)
    IF pao2 IS NOT NULL AND fio2 IS NOT NULL AND fio2 > 0 THEN
        pao2_fio2 := pao2 / fio2;
        IF pao2_fio2 < 100 THEN sofa_score := sofa_score + 4;
        ELSIF pao2_fio2 < 200 THEN sofa_score := sofa_score + 3;
        ELSIF pao2_fio2 < 300 THEN sofa_score := sofa_score + 2;
        ELSIF pao2_fio2 < 400 THEN sofa_score := sofa_score + 1;
        END IF;
    END IF;
    
    -- Coagulation (Platelets)
    IF platelets IS NOT NULL THEN
        IF platelets < 20 THEN sofa_score := sofa_score + 4;
        ELSIF platelets < 50 THEN sofa_score := sofa_score + 3;
        ELSIF platelets < 100 THEN sofa_score := sofa_score + 2;
        ELSIF platelets < 150 THEN sofa_score := sofa_score + 1;
        END IF;
    END IF;
    
    -- Liver (Bilirubin)
    IF bilirubin IS NOT NULL THEN
        IF bilirubin >= 12.0 THEN sofa_score := sofa_score + 4;
        ELSIF bilirubin >= 6.0 THEN sofa_score := sofa_score + 3;
        ELSIF bilirubin >= 2.0 THEN sofa_score := sofa_score + 2;
        ELSIF bilirubin >= 1.2 THEN sofa_score := sofa_score + 1;
        END IF;
    END IF;
    
    -- Cardiovascular (MAP)
    IF map IS NOT NULL THEN
        IF map < 70 THEN sofa_score := sofa_score + 1;
        END IF;
    END IF;
    
    -- Central nervous system (GCS)
    IF gcs IS NOT NULL THEN
        IF gcs < 6 THEN sofa_score := sofa_score + 4;
        ELSIF gcs < 10 THEN sofa_score := sofa_score + 3;
        ELSIF gcs < 13 THEN sofa_score := sofa_score + 2;
        ELSIF gcs < 15 THEN sofa_score := sofa_score + 1;
        END IF;
    END IF;
    
    -- Renal (Creatinine)
    IF creatinine IS NOT NULL THEN
        IF creatinine >= 5.0 THEN sofa_score := sofa_score + 4;
        ELSIF creatinine >= 3.5 THEN sofa_score := sofa_score + 3;
        ELSIF creatinine >= 2.0 THEN sofa_score := sofa_score + 2;
        ELSIF creatinine >= 1.2 THEN sofa_score := sofa_score + 1;
        END IF;
    END IF;
    
    RETURN sofa_score;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- VIEWS - Easy access to calculated fields
-- ============================================================================

CREATE OR REPLACE VIEW v_hourly_data_with_calculated AS
SELECT 
    h.id,
    h.patient_id,
    h.hour,
    h.vital_signs,
    h.vital_signs_extended,
    h.prediction,
    h.risk_level,
    h.timestamp,
    
    -- Extract core vitals from JSON
    (vital_signs->>'HR')::FLOAT as hr,
    (vital_signs->>'SBP')::FLOAT as sbp,
    (vital_signs->>'MAP')::FLOAT as map,
    
    -- Extract new biomarkers
    (vital_signs_extended->'sepsis_markers'->>'PCT')::FLOAT as pct,
    (vital_signs_extended->'sepsis_markers'->>'CRP')::FLOAT as crp,
    (vital_signs_extended->'sepsis_markers'->>'IL6')::FLOAT as il6,
    
    -- Calculate derived metrics
    calculate_shock_index(
        (vital_signs->>'HR')::FLOAT,
        (vital_signs->>'SBP')::FLOAT
    ) as shock_index,
    
    (vital_signs_extended->'hematology'->>'NLR')::FLOAT as nlr,
    (vital_signs_extended->'chemistry'->>'AnionGap')::FLOAT as anion_gap,
    (vital_signs_extended->'scores'->>'SOFA')::INT as sofa_score
    
FROM hourly_data h;

-- ============================================================================
-- REFERENCE RANGES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS biomarker_references (
    id SERIAL PRIMARY KEY,
    biomarker_name VARCHAR(50) UNIQUE NOT NULL,
    normal_min FLOAT,
    normal_max FLOAT,
    unit VARCHAR(20),
    category VARCHAR(50),
    clinical_significance TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert reference ranges
INSERT INTO biomarker_references (biomarker_name, normal_min, normal_max, unit, category, clinical_significance) VALUES
-- Sepsis Markers
('PCT', 0, 0.5, 'ng/mL', 'Sepsis Marker', 'Procalcitonin >0.5 suggests bacterial infection, >2.0 suggests sepsis'),
('CRP', 0, 10, 'mg/L', 'Inflammation', 'C-Reactive Protein elevation indicates inflammation'),
('Presepsin', 0, 600, 'pg/mL', 'Sepsis Marker', 'Presepsin >600 suggests sepsis'),
('IL6', 0, 7, 'pg/mL', 'Cytokine', 'IL-6 elevation indicates cytokine storm'),
('IL1b', 0, 5, 'pg/mL', 'Cytokine', 'IL-1β elevation indicates inflammation'),

-- Hematology
('ESR', 0, 20, 'mm/hr', 'Hematology', 'Elevated ESR indicates inflammation or infection'),
('NLR', 0.78, 3.53, 'ratio', 'Calculated', 'NLR >3.53 associated with poor prognosis'),
('PLR', 0, 180, 'ratio', 'Calculated', 'Platelet-Lymphocyte Ratio'),
('MPV', 7.5, 11.5, 'fL', 'Hematology', 'Mean Platelet Volume'),
('RDW', 11.5, 14.5, '%', 'Hematology', 'Red Cell Distribution Width'),

-- Coagulation
('D-Dimer', 0, 0.5, 'μg/mL', 'Coagulation', 'Elevated in DIC and thrombosis'),
('Fibrinogen', 200, 400, 'mg/dL', 'Coagulation', 'Decreased in DIC'),
('PT', 11, 13.5, 'seconds', 'Coagulation', 'Prothrombin Time'),
('INR', 0.8, 1.2, 'ratio', 'Coagulation', 'International Normalized Ratio'),

-- Chemistry
('Anion Gap', 8, 16, 'mEq/L', 'Chemistry', 'Elevated in metabolic acidosis'),
('Ionized Calcium', 1.16, 1.32, 'mmol/L', 'Electrolyte', 'Hypocalcemia common in sepsis'),
('Magnesium', 1.7, 2.4, 'mg/dL', 'Electrolyte', 'Hypomagnesemia in critical illness'),
('Albumin', 3.5, 5.5, 'g/dL', 'Protein', 'Hypoalbuminemia in sepsis'),

-- Scores
('SOFA', 0, 24, 'score', 'Clinical Score', 'Sequential Organ Failure Assessment, >2 indicates organ dysfunction'),
('qSOFA', 0, 3, 'score', 'Clinical Score', 'Quick SOFA, ≥2 suggests sepsis'),
('Shock Index', 0.5, 0.7, 'ratio', 'Calculated', 'HR/SBP, >0.9 indicates shock')
ON CONFLICT (biomarker_name) DO NOTHING;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE biomarker_references IS 'Reference ranges for all biomarkers with clinical significance';
COMMENT ON COLUMN hourly_data.vital_signs_extended IS 'Extended biomarker data in structured JSON format';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- To rollback: DROP TABLE hourly_data; ALTER TABLE hourly_data_backup RENAME TO hourly_data;
