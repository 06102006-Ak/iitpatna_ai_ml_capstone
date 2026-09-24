# Zepto AI/ML Capstone — End-to-End Analytics Platform

A single repository containing the three required modules:

- `/data_pipeline`
- `/analytics`
- `/support_assistant`

The project implements an end-to-end workflow covering data collection and database loading, exploratory analysis and machine learning, and a retrieval-augmented support assistant.

---

## Repository Map

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
│   ├── docs/
│   │   ├── doc_01.txt
│   │   └── ... doc_08.txt
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