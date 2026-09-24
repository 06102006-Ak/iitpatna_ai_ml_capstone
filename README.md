# Zepto AI/ML Capstone — End-to-End Analytics Platform

A single repository containing the three required modules: `/data_pipeline`, `/analytics`, and `/support_assistant`. The submission follows the specification's 100-mark structure and keeps the modules internally connected while allowing each folder to be graded independently.

## Repository map

```text
zepto-ai-ml-capstone/
├── data_pipeline/
│   ├── scrape_books.py
│   ├── load_and_query.py
│   ├── run_pipeline.py
│   ├── data/
│   ├── sql/
│   ├── outputs/
│   └── README.md
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── reload_check.py
│   ├── titanic.csv
│   ├── models/
│   ├── outputs/
│   └── README.md
├── support_assistant/
│   ├── docs/doc_01.txt ... doc_08.txt
│   ├── ingest.py
│   ├── graph.py
│   ├── prompt.py
│   ├── main.py
│   ├── run_examples.py
│   ├── Dockerfile
│   └── README.md
├── tests/
├── requirements.txt
└── README.md
```

## Setup

This repository uses **one consolidated `requirements.txt`** for all modules.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the full project

### 1) Data pipeline

Graded/live path — scrape the first five Books to Scrape catalogue pages:

```bash
python data_pipeline/run_pipeline.py
```

Offline-only deterministic smoke path for network-isolated development:

```bash
python data_pipeline/run_pipeline.py --offline-fixture
```

The required baseline conversion is **1 GBP = 105.50 INR**, a fixed project-defined constant. No currency API is needed.

Outputs include:
- cleaned CSV and data-quality JSON under `data_pipeline/data/` and `data_pipeline/outputs/`
- normalized SQLite schema generated from scratch
- six SQL query files under `data_pipeline/sql/`
- query transcripts in `data_pipeline/outputs/query_outputs.md`
- `pd.read_sql(...)` and `pd.merge(...)` equivalence evidence in `data_pipeline/outputs/pandas_validation.md`

### 2) Analytics pipeline

Run these in order:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/reload_check.py
```

`01_eda.py` contains the one required `sns.load_dataset('titanic')` call and immediately writes `analytics/titanic.csv`. `02_modeling.py` reads that committed CSV and never reloads the raw dataset from Seaborn.

The EDA report covers missingness thresholds, IQR outliers, fare mean/median/mode, survival-rate breakdowns, the exact six-column correlation matrix, strongest correlation pairs, at least four multivariate charts with written interpretations, and the z-score sanity check.

The modeling report covers a stratified split, train-only preprocessing via `ColumnTransformer`, Logistic Regression, Decision Tree, Random Forest, full classification metrics, ROC/AUC, imbalance comparison (baseline/class-weight/SMOTE), Random Forest `GridSearchCV` with OOB score, fare regression, adjusted R², residual heteroscedasticity analysis, a separate classification/regression metric presentation, and deployment-model selection.

### 3) Support assistant

First build the persistent vector index:

```bash
python -m support_assistant.ingest
```

Then run FastAPI:

```bash
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

The default `MOCK_LLM` state is enabled unless explicitly set to `0`. The graded baseline therefore makes no LLM API calls. The embedding/index layer is local using `all-MiniLM-L6-v2` + ChromaDB.

Example retrieval request:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"How much is standard delivery?"}'
```

Example general request:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the capital of India?"}'
```

## Architecture choices

**Data engineering:** scrape → parse → clean → fixed-rate conversion → normalized SQLite → SQL analytics → pandas equivalence check. The pipeline is idempotent and records data-quality decisions.

**Analytics:** load once → profile → clean with threshold policy → EDA/data story → exploratory standardization → stratified split → train-only preprocessing → three classifiers → imbalance/tuning analysis → fare regression → complete model artifact + reload check.

**GenAI/RAG:** corpus ingestion → per-document chunking → local embedding → ChromaDB → LangGraph intent router → top-3 cosine retrieval for policy questions → deterministic mock answer or optional real-LLM grounded generation → Pydantic response → FastAPI. See `support_assistant/README.md` for the stage-by-stage flow.

## Advanced engineering additions

The baseline rubric is preserved exactly; the extra engineering is additive:

- HTTP retries, timeouts, explicit status handling, de-duplication, and scrape statistics.
- Data-quality JSON audit output and deterministic offline fixture for development.
- SQLite indexes, constraints, foreign-key enforcement, and transactional/idempotent rebuild.
- Reproducible modeling with a fixed random seed, permutation importance, explicit technical model-selection rule, and raw-input reload check.
- RAG provenance metadata (`document_id`, `chunk_index`, source filename), persistent ChromaDB storage, health endpoint, environment-controlled LLM backend, structured-output retries, and Docker packaging.
- Automated tests that validate core parsing, schema decisions, and repository requirements without contacting external services.

## Git workflow evidence

The repository is intended to be submitted with a visible feature branch that has at least two commits and a merge commit back into `main`. The provided project archive includes that history so `git log --graph --all` shows the required branch/merge workflow.

## Academic-integrity note

The implementation is purpose-built for the assignment specification. Generated repository files should be reviewed, understood, and adapted by the student before submission, especially any written interpretation or model-selection wording that becomes part of the student's final explanation.
