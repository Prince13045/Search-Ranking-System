<img width="1907" height="976" alt="Screenshot 2026-08-01 124638" src="https://github.com/user-attachments/assets/e328829d-9c3a-4b4d-a6d1-59f289011c81" />
## 🏗️ System Architecture

```mermaid
flowchart TD

A[User Search Query] --> B[Streamlit UI]

B --> C[FastAPI API]

C --> D[Prediction Pipeline]

D --> E[TF-IDF Candidate Retrieval]

E --> F[Top 100 Candidate Products]

F --> G[Feature Engineering]

G --> H[Preprocessor]

H --> I[XGBoost Ranker]

I --> J[Ranked Products]

J --> K[FastAPI Response]

K --> L[Streamlit UI]

L --> M[Top 10 Products Display]
```

## 🧠 Machine Learning Pipeline

```mermaid
flowchart LR

A[MySQL Database]

A --> B[Data Ingestion]

B --> C[Data Validation]

C --> D[Feature Engineering]

D --> E[TF-IDF Similarity]

E --> F[Column Transformer]

F --> G[Train Test Split]

G --> H[XGBRanker]

H --> I[Model Evaluation]

I --> J[Artifacts]

J --> K[xgb_ranker.pkl]

J --> L[preprocessor.pkl]

J --> M[tfidf_vectorizer.pkl]
```

## 📂 Project Structure

```mermaid
graph TD

A[Search Ranking System]

A --> B[src]

A --> C[data]

A --> D[artifacts]

A --> E[config]

A --> F[Streamlit]

A --> G[FastAPI]

B --> H[components]

B --> I[pipeline]

H --> J[data_ingestion.py]

H --> K[data_validation.py]

H --> L[feature_engineering.py]

H --> M[model_training.py]

H --> N[prediction.py]

I --> O[training_pipeline.py]

I --> P[prediction_pipeline.py]
```
## 🚀 Deployment Architecture

```mermaid
flowchart LR

A[Browser]

A --> B[Streamlit]

B --> C[FastAPI]

C --> D[XGBRanker]

D --> E[Artifacts]

E --> F[Model]

E --> G[Preprocessor]

E --> H[TF-IDF Vectorizer]
```
## 🔍 Search Ranking Flow

```mermaid
sequenceDiagram

participant User

participant Streamlit

participant API

participant Retrieval

participant Ranker

User->>Streamlit: Search "gaming laptop"

Streamlit->>API: POST /predict

API->>Retrieval: Retrieve Top 100 Products

Retrieval-->>API: Candidate Products

API->>Ranker: Compute Features

Ranker-->>API: Ranking Scores

API-->>Streamlit: Top 10 Products

Streamlit-->>User: Display Results
```
# 🛒 Search Ranking System

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49-red?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Learning--to--Rank-orange)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![MySQL](https://img.shields.io/badge/MySQL-Database-blue?logo=mysql)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An end-to-end **Amazon-style Search Ranking System** that retrieves candidate products using TF-IDF and ranks them using **XGBoost Learning-to-Rank**, exposed through **FastAPI** and an interactive **Streamlit** interface.

<img width="1916" height="974" alt="serach ranking syste4m (1)" src="https://github.com/user-attachments/assets/3eb2ac1d-18cd-4cf7-8794-7fcb2209b6d5" />

