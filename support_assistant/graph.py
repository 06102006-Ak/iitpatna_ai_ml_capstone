from __future__ import annotations

import json
import os
from typing import TypedDict

import chromadb
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph
import requests

from .ingest import CHROMA_DIR, COLLECTION_NAME, MODEL_NAME
from .prompt import DIRECT_PROMPT, PROMPT_TEMPLATE

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
GENERAL_MOCK = "I can only answer questions about Zepto policies right now."


class SupportResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_documents: list[str]
    retrieved_ids: list[str]
    distances: list[float]
    answer: str
    response: SupportResponse


def mock_enabled() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


# Lazily initialized so importing the FastAPI app does not download the embedding model.
_embedding_model: SentenceTransformer | None = None
_chroma_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _chroma_collection


def call_real_llm(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required when MOCK_LLM=0")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def classify_intent(state: GraphState) -> dict:
    query = state["query"]
    lower = query.lower()
    if mock_enabled():
        intent = "policy_question" if any(keyword in lower for keyword in POLICY_KEYWORDS) else "general_question"
    else:
        prompt = f"Classify this query as exactly one label: policy_question or general_question. Query: {query}"
        raw = call_real_llm(prompt).strip().lower()
        intent = "policy_question" if "policy_question" in raw else "general_question"
    return {"intent": intent}


def parse_valid_response(raw: str, *, retries: int = 2, corrective_context: str = "") -> SupportResponse:
    last_error: Exception | None = None
    candidate = raw
    for attempt in range(retries + 1):
        try:
            data = json.loads(candidate)
            return SupportResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            corrective = corrective_context or "Return only valid JSON with answer, sources, confidence; confidence must be between 0 and 1."
            candidate = call_real_llm(corrective + "\nPrevious invalid output:\n" + candidate)
    return SupportResponse(answer=f"[ERROR] Structured response validation failed after {retries + 1} attempts: {last_error}", sources=[], confidence=0.0)


def retrieve_top3(query: str) -> tuple[list[str], list[str], list[float]]:
    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = get_collection().query(query_embeddings=[query_embedding], n_results=3, include=["documents", "metadatas", "distances"])
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = [float(x) for x in results.get("distances", [[]])[0]]
    ids = []
    for md in metadatas:
        ids.append(f"{md['document_id']}_chunk_{int(md['chunk_index']):02d}")
    return documents, ids, distances


def retrieve_and_answer(state: GraphState) -> dict:
    documents, ids, distances = retrieve_top3(state["query"])
    if not documents:
        return {"retrieved_documents": [], "retrieved_ids": [], "distances": [], "answer": "[ERROR] No policy context was retrieved." , "response": SupportResponse(answer="[ERROR] No policy context was retrieved.", sources=[], confidence=0.0)}
    if mock_enabled():
        snippet = documents[0][:200].strip()
        response = SupportResponse(answer=f"Based on the retrieved context: {snippet}", sources=ids, confidence=1.0)
    else:
        context = "\n\n".join(f"[{doc_id}] {doc}" for doc_id, doc in zip(ids, documents))
        prompt = PROMPT_TEMPLATE.format(context=context, question=state["query"])
        raw = call_real_llm(prompt)
        response = parse_valid_response(raw, corrective_context=prompt + "\nCORRECTION: Previous output failed validation. Return only the required JSON object.")
        if not response.sources:
            response.sources = ids
    return {"retrieved_documents": documents, "retrieved_ids": ids, "distances": distances, "answer": response.answer, "response": response}


def direct_answer(state: GraphState) -> dict:
    if mock_enabled():
        response = SupportResponse(answer=GENERAL_MOCK, sources=[], confidence=1.0)
    else:
        prompt = DIRECT_PROMPT.format(question=state["query"])
        raw = call_real_llm(prompt)
        response = parse_valid_response(raw, corrective_context=prompt + "\nCORRECTION: Return only valid JSON with answer, sources, confidence.")
        response.sources = []
    return {"answer": response.answer, "response": response}


def route_intent(state: GraphState) -> str:
    return state["intent"]


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {"policy_question": "retrieve_and_answer", "general_question": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


app_graph = build_graph()


def ask(query: str) -> SupportResponse:
    result = app_graph.invoke({"query": query})
    response = result.get("response")
    if isinstance(response, SupportResponse):
        return response
    return SupportResponse.model_validate(response)
