# Customer Churn Prediction and Lifecycle Management Framework

Implementation of the framework specified in *A Framework for Customer Churn
Prediction and Lifecycle Management to Curtail Customer Attrition in Nigerian
Digital Banks* — Chapters 1 to 3.

Every component traces back to a specific section of Chapter 3:

| Component                | Implements                          | Chapter 3 ref |
|--------------------------|-------------------------------------|---------------|
| `src/preprocessing.py`   | Data preprocessing pipeline         | Algorithm 3.1 |
| `src/modeling.py`        | Four-model training + selection     | Algorithm 3.2 |
| `src/evaluation.py`      | Five-metric evaluation              | §3.4.2        |
| `src/explainability.py`  | SHAP global + local                 | §3.4.2, Eq. 3.13 |
| `src/segmentation.py`    | Risk-tier assignment                | Algorithm 3.2, step 6 |
| `src/recommender.py`     | Retention recommendation rules      | Algorithm 3.2, step 7 |
| `api/main.py`            | REST API (application layer)        | §3.5.1 |
| `dashboard/app.py`       | Streamlit dashboard (5 tabs)        | §3.5.4 |

---

## 1. Prerequisites

- **Python 3.10 or higher** (Chapter 3, Table 3.4)
- **pip** (bundled with Python)
- **git** (optional, but recommended for version control)

Check what you have:

```bash
python --version       # should be 3.10+
pip --version
```

On Windows you may need to use `python` or `py`; on macOS/Linux you may need
`python3` and `pip3`.

---

## 2. One-time setup

Open a terminal in the project root (`churn-lifecycle-mgmt/`).

### 2.1 Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your prompt once activated.

### 2.2 Upgrade pip and install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes the first time — `shap`, `xgboost`, and `lightgbm`
are the heavier installs.

### 2.3 Place the dataset

Copy `Churn_Modelling_Engineered.csv` into `data/raw/`. The pipeline expects
this exact filename.

### 2.4 Configure the database (PostgreSQL via Neon)

Chapter 3 §3.5.1 commits the data layer to PostgreSQL v15+. We use
[Neon](https://neon.tech) — a managed Postgres service with a free tier
that's more than enough for the 10,000-record dataset.

1. Sign in to Neon and create a new project (region: closest to you).
2. In the dashboard, open **Connection Details** and choose the
   **SQLAlchemy** snippet. Copy the connection string.
3. In the project root, copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env       # macOS/Linux
   copy .env.example .env     # Windows
   ```

4. Open `.env` and paste your connection string into `DATABASE_URL`.
   Keep the `postgresql+psycopg://` prefix — that tells SQLAlchemy to use
   the psycopg v3 driver.

The schema (tables for customers, predictions, SHAP values, recommendations,
and model metadata) is created automatically the first time the pipeline runs.

### 2.5 Verify the install

```bash
python -c "import pandas, sklearn, xgboost, lightgbm, shap, imblearn; print('OK')"
```

If you see `OK`, the Python packages are in. Next, check the database connection:

```bash
python -c "from sqlalchemy import create_engine, text; from src.config import DATABASE_URL; \
print(create_engine(DATABASE_URL).connect().execute(text('SELECT 1')).scalar())"
```

That should print `1`. If both checks pass, we're ready.

---

## 3. Running the pipeline (later steps)

Once each module is built we'll run:

```bash
python scripts/run_pipeline.py        # end-to-end: preprocess → train → evaluate → SHAP → segment → recommend
uvicorn api.main:app --reload         # FastAPI backend on http://localhost:8000
streamlit run dashboard/app.py        # Streamlit dashboard on http://localhost:8501
```

---

## 4. Project layout

```
churn-lifecycle-mgmt/
├── data/raw/                  Original dataset (Kaggle Bank Customer Churn)
├── data/processed/            Splits and scaled arrays
├── models/                    Serialised model + scaler (churn_model.joblib)
├── outputs/figures/           ROC curves, confusion matrices, SHAP plots
├── outputs/tables/            Model-comparison and segmentation tables (CSV)
├── src/                       Core ML pipeline modules
├── api/                       FastAPI application layer
├── dashboard/                 Streamlit presentation layer
├── scripts/run_pipeline.py    Reproduces every Chapter 4 result in one run
├── requirements.txt
└── README.md
```

---

## 5. Reproducibility

`random_state=42` is fixed across the train/test split, SMOTE, and every model,
per Chapter 3, §3.3.4. Re-running `scripts/run_pipeline.py` from a clean clone
should reproduce every figure and table reported in Chapter 4.
