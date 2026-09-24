# Module 3 — Grounded Zepto Support Assistant

## Required baseline
The graded path is deterministic and offline at the LLM layer. `MOCK_LLM` is treated as enabled unless explicitly set to `0`. Embeddings are generated locally with `sentence-transformers` using `all-MiniLM-L6-v2`, and vectors are stored in ChromaDB.

## Build the corpus index
```bash
python support_assistant/ingest.py
```

This creates a persistent ChromaDB collection named `zepto_policy` under `support_assistant/chroma_db/`. The corpus contains exactly the 8 assignment documents under `docs/`, with one chunk per document by default.

## Run the FastAPI service
```bash
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

### Required mock-mode examples
With `MOCK_LLM` unset (default behavior):

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How much is standard delivery?"}'
```

Expected structure:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials...","sources":["doc_01_chunk_00"],"confidence":1.0}
```

And a general query:

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of India?"}'
```

Expected structure:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The exact first ~200 characters of the retrieved chunk can vary only with chunking/index configuration; the code itself is deterministic in mock mode after the embedding model is fixed.

## Architecture: ingestion → embedding → retrieval → generation

```text
8 policy .txt files
      |
      v
ingest.py: read + chunk per document
      |
      v
SentenceTransformer(all-MiniLM-L6-v2)
      |
      v
ChromaDB collection: zepto_policy
      |
      v
LangGraph StateGraph
  classify_intent
      |---- policy_question ----> retrieve_and_answer
      |                              |
      |                              v
      |                        top-3 cosine retrieval
      |                              |
      |                              v
      |                        answer generation
      |
      |---- general_question ----> direct_answer

Final node -> Pydantic SupportResponse(answer, sources, confidence) -> FastAPI /ask
```

`ingest.py` owns ingestion, chunking, local embedding, and ChromaDB persistence. `retrieve_and_answer` in `graph.py` embeds the query and performs top-3 cosine retrieval; retrieval is always real in both mock and real-LLM modes. `retrieve_and_answer` then either creates the required mock response from the top chunk or uses the structured role–context–task–format–length prompt for the optional real LLM. `direct_answer` similarly uses the fixed mock response or an optional direct LLM call.

The `MOCK_LLM` toggle affects generation/classification only:
- Default/unset or `1`: keyword intent heuristic, deterministic canned answers, no LLM network call.
- `0`: optional Groq-compatible HTTP call using `GROQ_API_KEY`, with structured-output validation and up to two corrective retries on invalid output.

## Prompt template
The exact prompt text is in `prompt.py`. It explicitly includes role, context, task, format, length, a negative constraint, and a few-shot example. It is used only on the optional `MOCK_LLM=0` branch, as required.

## Docker
Build and run locally:

```bash
docker build -t zepto-support -f support_assistant/Dockerfile .
docker run --rm -p 7860:7860 zepto-support
```

The image starts Uvicorn on `0.0.0.0:7860`. Build-time model downloads may require outbound network in the environment because `sentence-transformers` needs the embedding model unless the image is prepared with a pre-cached model.
