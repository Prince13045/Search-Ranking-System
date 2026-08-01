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
