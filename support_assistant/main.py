from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import SupportResponse, ask

app = FastAPI(title="Zepto Grounded Support Assistant", version="1.0.0")


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=SupportResponse)
def ask_endpoint(request: AskRequest) -> SupportResponse:
    try:
        return ask(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
