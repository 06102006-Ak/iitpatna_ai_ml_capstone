# Module 1 — Data Pipeline

## Scope
Scrape books from Books to Scrape using `requests` + `BeautifulSoup`, clean the fields, convert GBP to INR using the assignment's fixed `1 GBP = 105.50 INR`, load a normalized SQLite database, and validate SQL/pandas equivalence.

## Run
From the repository root:

```bash
python data_pipeline/run_pipeline.py
```

The default path scrapes the first five listing pages (100 books before de-duplication). For offline smoke-testing only, use:

```bash
python data_pipeline/run_pipeline.py --offline-fixture
```

The live path is the graded path. The offline fixture exists only because some development environments have no network access.

## Design decisions
- `requests.Session` + retries/backoff handle transient HTTP failures; non-2xx responses are raised explicitly.
- Detail pages are queried for the category so the final dataset contains a true category value rather than a hard-coded label.
- Numeric parse failures are median-imputed after the clean numeric series is computed; rows with missing title/category or unparseable categorical fields are dropped because those fields are required identifiers for relational storage.
- `price_inr = price_gbp * 105.50` exactly, with no market-rate lookup.
- SQLite enables foreign keys and uses `categories` → `books` as the normalized primary/foreign-key relationship.
- The pipeline is idempotent: each full run recreates the database schema from scratch.
- A data-quality report records row counts and parse failures so the cleaning decisions are auditable.

## SQL + pandas evidence
`run_pipeline.py` writes six SQL query files under `sql/`, executes them against SQLite, writes their outputs to `outputs/query_outputs.md`, and verifies the JOIN result against an in-memory `pd.merge(...)` reproduction.
