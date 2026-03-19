# Bank Transaction Fraud Detection System

End-to-end fraud detection system using ensemble ML models with real-time API inference and LLM-powered explanations.

## Project Overview

Developed for Devoteam ML Engineer Interview:
- **Exploratory Data Analysis**: Transaction pattern analysis
- **Feature Engineering**: Behavioral, temporal, account-level features
- **Ensemble Model**: Isolation Forest + One-Class SVM + LOF + Autoencoder
- **Advanced Model**: Behavioral Fingerprinting with Attention (Original)
- **Production API**: FastAPI with async processing
- **LLM Integration**: Google Vertex AI for fraud explanations

## Dataset

- **Source**: [Bank Transaction Dataset (Kaggle)](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection/data)
- **Size**: 2,512 transactions, 16 features

## Quick Start

### Prerequisites

- Python 3.11+
- UV (recommended) or pip

### Installation

```bash
cd devoteam

# UV (recommended)
pip install uv
uv sync

# Or pip
pip install -r requirements.txt
```

### Running

```bash
# Run notebooks
jupyter notebook
# Execute: 01_eda.ipynb -> 02_modeling.ipynb -> 03_behavioral_fingerprinting.ipynb

# Run API
uvicorn main:app --reload
# API docs: http://localhost:8000/docs

# Docker (optional)
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

## Repository Structure

```
devoteam/
├── bank_transactions_data.csv       # Raw dataset
├── config.py                         # Configuration
├── main.py                           # FastAPI application
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA + Business Insights
│   ├── 02_modeling.ipynb             # Ensemble Baseline
│   └── 03_behavioral_fingerprinting.ipynb  # Advanced Model
├── src/
│   ├── features.py                   # Feature Engineering
│   ├── models.py                     # Ensemble Detector
│   ├── inference.py                  # Model Inference
│   ├── behavioral_model.py           # Behavioral Fingerprinting
│   ├── visualization.py              # Custom Visualizations
│   └── llm_explainer.py              # Vertex AI Explanations
├── api/
│   └── models.py                     # Pydantic Schemas
├── models/                           # Trained models
├── utils/
│   └── gemini_client.py              # Gemini client utilities
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/stats` | GET | API statistics |
| `/api/v1/predict` | POST | Single transaction prediction |
| `/api/v1/predict/batch` | POST | Batch prediction |
| `/api/v1/account/{id}/risk` | GET | Account risk profile |

## Configuration

Create `.env` file:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_PATH=models/fraud_detector.pkl
FEATURES_PATH=models/features.txt
HISTORICAL_DATA_PATH=bank_transactions_data.csv

# LLM (optional)
ENABLE_LLM_EXPLANATIONS=true
PROJECT_ID=your-gcp-project-id
LOCATION=us-central1
CREDENTIALS_FILE_PATH=path/to/service-account.json

# Redis (optional)
REDIS_ENABLED=false
```

## Key Features

### Ensemble Anomaly Detection

- **Isolation Forest** (30%): Efficient anomaly isolation
- **One-Class SVM** (30%): Complex boundary learning
- **LOF** (20%): Density-based detection
- **Autoencoder** (20%): Neural reconstruction error

### Behavioral Fingerprinting (Original)

- **Attention Mechanism**: Learns important features per prediction
- **Per-Account Thresholds**: Adaptive baselines (fairer)
- **Multi-Format Export**: TorchScript + ONNX

### Production Features

- Async processing
- Redis caching
- Pydantic validation
- OpenAPI/Swagger docs

## Deliverables

1. **Notebooks**: EDA, Baseline, Advanced Model
2. **Source Code**: Full implementation
3. **API**: Working FastAPI with Swagger docs
4. **Docker**: Optional container support

---

**Developed for**: Devoteam ML Engineer Interview
**Date**: March 2026
**Tech Stack**: Python, FastAPI, PyTorch, Vertex AI
