from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scholaragent import config as cfg
from scholaragent import tools
from scholaragent.agent import compare_with_llm, run_agent

app = FastAPI(title="ScholarAgent", version="0.1.0", description="Agentic RAG over arXiv papers")

class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    history: list[Message] = []

class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(5, ge=1, le=20)

class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=4)
    aspect: str = Field(min_length=3)

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": f"{cfg.LLM_PROVIDER}:{cfg.LLM_MODEL or 'default'}", "embed_model": cfg.EMBED_MODEL}

@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return run_agent(req.question, [m.model_dump() for m in req.history])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.post("/search")
def search(req: SearchRequest) -> list[dict]:
    try:
        return tools.search_papers(req.query, req.k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/papers/{paper_id:path}")
def paper(paper_id: str) -> dict:
    p = tools.get_paper(paper_id)
    if p is None:
        raise HTTPException(status_code=404, detail="paper not in index")
    return p

@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    return compare_with_llm(req.paper_ids, req.aspect)