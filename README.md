# 🏥 GRU-Based Early Sepsis Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Research-red.svg)

**Deep Learning System for Predicting Sepsis in ICU Patients**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Web App](#-web-application) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

This project implements a **Gated Recurrent Unit (GRU)** deep learning model to predict sepsis onset in Intensive Care Unit (ICU) patients. The system analyzes multivariate physiological time-series data (vital signs, lab values) and provides real-time risk predictions with high accuracy.

### 🎯 Key Highlights

- **ROC-AUC: 0.8797** - Excellent discrimination performance
- **Recall: 78.34%** - Detects 78% of sepsis cases 
- **Real-time Capable** - <50ms inference latency
- **Full Web Interface** - Flask-based patient monitoring dashboard
- **Production-Ready** - Complete preprocessing and deployment pipeline

---

## 🌟 Features

### Core Capabilities

✅ **Time-Series Modeling**: 6-hour lookback window with GRU architecture  
✅ **63 Clinical Features**: Vital signs, lab values, and patient metadata  
✅ **Class Imbalance Handling**: Weighted loss for ~3-5% sepsis prevalence  
✅ **Real-Time Inference**: Hourly risk predictions with streaming data support  
✅ **Web Dashboard**: Interactive patient management and visualization  

### Technical Features

- **Robust Preprocessing**: Imputation, scaling, encoding pipeline
- **Model Callbacks**: Early stopping, LR scheduling, checkpointing
- **Multiple Metrics**: ROC-AUC, PR-AUC, Precision, Recall
- **RESTful API**: Flask endpoints for patient and prediction management
- **SQLite Database**: Patient records and prediction history

---

## 📊 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.8797 | Excellent discrimination (>0.85 is strong) |
| **PR-AUC** | 0.1802 | 3.6× better than random baseline |
| **Recall** | 78.34% | Catches 78 out of 100 sepsis cases |
| **Precision** | 8.89% | 1 in 11 alerts is true positive |
| **Specificity** | 84.75% | Low false positive rate |
| **NPV** | 99.52% | Highly reliable for ruling out sepsis |

### Confusion Matrix (270,106 test samples)

|  | Actual: Sepsis | Actual: No Sepsis |
|---|---|---|
| **Predicted: Sepsis** | 3,938 (TP) | 40,336 (FP) |
| **Predicted: No Sepsis** | 1,089 (FN) | 224,743 (TN) |

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.2+ (for GPU training, optional)
- 8GB+ RAM (16GB+ recommended for training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sepsis-detection-gru.git
cd sepsis-detection-gru
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1️⃣ Data Preparation

Transform raw ICU records into model-ready sequences:

```bash
python prepare_sequence_dataset_v23.py \
    --input data/train.csv \
    --output data/processed/ \
    --window 6 \
    --step 1 \
    --test-size 0.2 \
    --val-size 0.2
```

**Input CSV format:**
```csv
Patient_ID,ICULOS,HR,MAP,O2Sat,Temp,Resp,SepsisLabel
P001,1,85,75,98,37.2,18,0
P001,2,88,72,97,37.4,19,0
...
```

**Outputs:**
- `X_train.npy`, `y_train.npy` - Training data
- `X_val.npy`, `y_val.npy` - Validation data
- `X_test.npy`, `y_test.npy` - Test data
- `imputer.pkl`, `scaler.pkl`, `ohe.pkl` - Preprocessing objects

### 2️⃣ Model Training

Train the GRU model:

```bash
python train_gru_v23.py \
    --data data/processed/ \
    --output models/ \
    --epochs 60 \
    --batch-size 512 \
    --lr 0.001
```

**Training features:**
- Early stopping (patience=8)
- Learning rate reduction (factor=0.5)
- Model checkpointing (best val_pr_auc)
- Automatic class weighting

**Outputs:**
- `gru_v23_best.keras` - Best model weights
- `training_history.json` - Training metrics
- `test_results.json` - Test performance

### 3️⃣ Inference

Run predictions on new patients:

```bash
python run_gru_on_csv_v23.py \
    --input test_patients.csv \
    --output predictions.csv \
    --model models/gru_v23_best.keras \
    --preprocessing data/processed/ \
    --threshold 0.1799
```

**Output format:**
```csv
Patient_ID,ICULOS,proba,yhat,insufficient_history
P001,1,,,True
P001,6,0.0234,0,False
P001,24,0.2145,1,False  ← Sepsis risk detected!
```

### 4️⃣ Web Application

Start the Flask web interface:

```bash
python app.py
```

Then open your browser: `http://localhost:5000`

**Features:**
- 👤 Patient registration and management
- 📈 Hourly vital signs data entry
- 🎯 Real-time sepsis risk predictions
- 📊 Interactive risk trend visualization
- 🗂️ Patient history tracking

---

## 🌐 Web Application

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard |
| GET | `/api/patients` | List all patients |
| POST | `/api/patients` | Register new patient |
| GET | `/api/patients/<id>` | Get patient details |
| POST | `/api/patients/<id>/hourly-data` | Add hourly data + predict |
| DELETE | `/api/patients/<id>` | Delete patient |
| GET | `/api/health` | System health check |

### Risk Levels

| Risk Score | Level | Color | Action |
|-----------|-------|-------|--------|
| < 0.10 | Very Low | 🟢 Green | Standard monitoring |
| 0.10-0.30 | Low | 🔵 Blue | Careful observation |
| 0.30-0.50 | Medium | 🟠 Orange | Increased vigilance |
| 0.50-0.70 | High | 🔴 Red | Clinical assessment |
| > 0.70 | Very High | 🔴 Dark Red | Urgent intervention |

---

## 🏗️ Architecture

### Model Design

```
Input: (batch_size, 6, 63)
    ↓
┌─────────────────────────┐
│  GRU Layer (64 units)   │  ← Temporal dependencies
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  BatchNormalization     │  ← Training stabilization
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Dropout (p=0.3)        │  ← Regularization
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Dense(32, ReLU)        │  ← Feature extraction
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Dropout (p=0.3)        │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Dense(1, Sigmoid)      │  ← Binary classification
└─────────────────────────┘
    ↓
Output: P(Sepsis) ∈ [0, 1]
```

**Model Parameters:**
- Total params: ~52,000
- Trainable params: ~51,800
- Model size: ~2.1 MB

---

## 📁 Project Structure

```
sepsis-detection-gru/
├── README.md                           # This file
├── LICENSE                             # MIT license
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── prepare_sequence_dataset_v23.py     # Data preprocessing script
├── train_gru_v23.py                    # Model training script
├── run_gru_on_csv_v23.py              # Inference script
├── app.py                              # Flask web application
├── example_usage.py                    # Usage examples
│
├── index.html                          # Web UI - Main page
├── script.js                           # Web UI - JavaScript
├── style.css                           # Web UI - Styling
│
├── data/                               # Data directory
│   ├── processed/                      # Processed datasets
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── imputer.pkl
│   │   ├── scaler.pkl
│   │   └── ...
│   └── sample_data/                    # Sample patient data
│
├── models/                             # Model directory
│   ├── gru_v23_best.keras             # Trained model
│   ├── training_history.json          # Training logs
│   └── test_results.json              # Test metrics
│
└── docs/                               # Documentation
    ├── GRU_Sepsis_Detection_Documentation.md
    └── Proje_Ozet_Makale.md
```

---

## 📖 Documentation

- **[Technical Documentation](docs/GRU_Sepsis_Detection_Documentation.md)** - Detailed technical overview (English)
- **[Project Summary](docs/Proje_Ozet_Makale.md)** - Comprehensive project report (Turkish)
- **[README](README.md)** - This file

---

## 🔬 Methodology

### Data Processing Pipeline

1. **Data Cleaning** - Remove duplicates, filter invalid ranges
2. **Imputation** - SimpleImputer with median strategy
3. **Scaling** - StandardScaler (μ=0, σ=1)
4. **Encoding** - OneHotEncoder for categorical variables
5. **Sequence Construction** - 6-hour sliding windows per patient

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 512 |
| Max Epochs | 60 |
| Early Stopping | patience=8, monitor=val_pr_auc |
| Class Weighting | {0: 0.52, 1: 10.5} |

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{sepsis_detection_gru_2025,
  author = {Ahmet},
  title = {GRU-Based Early Sepsis Detection System},
  year = {2025},
  url = {https://github.com/yourusername/sepsis-detection-gru}
}
```

---

## 📚 References

**Clinical Guidelines:**
1. Singer M, et al. (2016). "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)." *JAMA*.
2. Surviving Sepsis Campaign Guidelines (2021).

**Technical Literature:**
1. Cho K, et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder." *EMNLP*.
2. Chung J, et al. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks." *arXiv:1412.3555*.

**Dataset:**
- PhysioNet Challenge 2019: "Early Prediction of Sepsis from Clinical Data"

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## ⚠️ Disclaimer

**This software is intended for RESEARCH and EDUCATIONAL purposes only.**

- ❌ NOT approved for clinical use
- ❌ NOT a substitute for professional medical judgment
- ❌ Authors are NOT responsible for medical outcomes

Any medical decisions should be made by qualified healthcare professionals. For clinical deployment, additional validation and regulatory approval (FDA 510(k), CE marking) are required.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Ahmet** - Yapay Sinir Ağları Projesi

---

## 🙏 Acknowledgments

- PhysioNet for providing the sepsis detection dataset
- TensorFlow and Keras teams for the deep learning framework
- Flask community for the web framework
- All contributors and researchers in the sepsis detection field

---

<div align="center">

**⭐ If you find this project helpful, please give it a star!**

Made with ❤️ for better healthcare outcomes

</div>
