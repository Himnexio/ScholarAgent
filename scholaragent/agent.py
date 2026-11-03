from __future__ import annotations
import operator
from functools import lru_cache
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph
from scholaragent import config as cfg
from scholaragent.llm import ask_json, ask_llm
from scholaragent.tools import compare_papers, get_retriever
from scholaragent.utils import check_citations

PLAN_PROMPT = """You help search a collection of academic papers.
Conversation so far:
{history}
New question: {question}
Write 1-3 standalone search queries (keywords or short phrases, no pronouns) that would retrieve the passages
needed to answer the new question. Use several queries only if the question has several distinct parts.
Return JSON: {{"queries": ["...", "..."]}}"""
GRADE_PROMPT = """Question: {question}
Retrieved passages:
{context}
Do these passages contain enough information to answer the question well?
Return JSON: {{"sufficient": true or false, "missing": "what is missing, if anything",
"better_query": "an improved search query if insufficient, otherwise an empty string"}}"""

ANSWER_SYSTEM = (
    "You are a careful research assistant. Answer ONLY from the numbered passages. Cite every claim with its "
    "passage number, like [1] or [2, 3]. If the passages do not contain the answer, say so and state what is "
    "missing. Never invent papers, numbers or citations."
)
ANSWER_PROMPT = """Conversation so far:
{history}
Passages:
{context}
Question: {question}
Answer clearly and concisely, with inline citations."""
COMPARE_PROMPT = """Compare the papers below on this aspect: {aspect}
{evidence}
Write a short markdown table (one row per paper) followed by a 2-3 sentence takeaway. Use only the evidence
above and mention the arXiv id of each paper. If evidence is missing for a paper, say so."""

class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    queries: list[str]
    docs: list[dict]
    attempts: int
    sufficient: bool
    better_query: str
    answer: str
    citations: list[dict]
    trace: Annotated[list[str], operator.add]

def format_history(history: list[dict], last: int = 4) -> str:
    return "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-last:])

def format_context(docs: list[dict], max_words: int = 260) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        text = " ".join(d["text"].split()[:max_words])
        blocks.append(f"[{i}] {d['title']} ({d['year']}), section: {d['section']}\n{text}")
    return "\n\n".join(blocks)

def plan_node(state: AgentState) -> dict:
    prompt = PLAN_PROMPT.format(history=format_history(state.get("history", [])) or "(none)", question=state["question"])
    try:
        raw = ask_json(prompt).get("queries", [])
        queries = [q.strip() for q in raw if isinstance(q, str) and q.strip()][:3]
    except Exception:
        queries = []
    queries = queries or [state["question"]]
    return {"queries": queries, "trace": [f"plan -> {queries}"]}

def retrieve_node(state: AgentState) -> dict:
    retriever = get_retriever()
    pool: dict[str, dict] = {}
    for q in state["queries"]:
        for d in retriever.search(q, k=cfg.CANDIDATES_PER_QUERY, rerank=False):
            if d["chunk_id"] not in pool or d["score"] > pool[d["chunk_id"]]["score"]:
                pool[d["chunk_id"]] = d
    candidates = sorted(pool.values(), key=lambda d: d["score"], reverse=True)[:25]
    docs = retriever.rerank(state["question"], candidates, cfg.CONTEXT_K) if cfg.USE_RERANKER else candidates[: cfg.CONTEXT_K]
    attempts = state.get("attempts", 0) + 1
    return {"docs": docs, "attempts": attempts,
            "trace": [f"retrieve (attempt {attempts}) -> {len(candidates)} candidates, {len(docs)} passages kept"]}

def grade_node(state: AgentState) -> dict:
    if not state.get("docs"):
        return {"sufficient": False, "better_query": state["question"], "trace": ["grade -> no passages found"]}
    try:
        out = ask_json(GRADE_PROMPT.format(question=state["question"], context=format_context(state["docs"], 120)))
        sufficient = bool(out.get("sufficient"))
        better = str(out.get("better_query") or "").strip()
    except Exception:
        sufficient, better = True, ""
    return {"sufficient": sufficient, "better_query": better,
            "trace": [f"grade -> {'sufficient' if sufficient else 'weak, will retry'}"]}

def route_after_grade(state: AgentState) -> str:
    if state.get("sufficient") or state.get("attempts", 0) >= cfg.MAX_ATTEMPTS:
        return "answer"
    return "rewrite"

def rewrite_node(state: AgentState) -> dict:
    new_q = state.get("better_query") or state["question"]
    return {"queries": [new_q], "trace": [f"rewrite -> {new_q!r}"]}

def answer_node(state: AgentState) -> dict:
    docs = state.get("docs", [])
    if not docs:
        return {"answer": "I couldn't find relevant passages in the indexed papers for that question."}
    prompt = ANSWER_PROMPT.format(
        history=format_history(state.get("history", [])) or "(none)",
        context=format_context(docs), question=state["question"],
    )
    return {"answer": ask_llm(prompt, system=ANSWER_SYSTEM, temperature=0.1), "trace": ["answer generated"]}

def cite_node(state: AgentState) -> dict:
    docs = state.get("docs", [])
    answer, used = check_citations(state.get("answer", ""), len(docs))
    citations = [
        {"n": n, "paper_id": docs[n - 1]["paper_id"], "title": docs[n - 1]["title"], "authors": docs[n - 1]["authors"],
         "year": docs[n - 1]["year"], "section": docs[n - 1]["section"], "url": docs[n - 1]["url"],
         "snippet": " ".join(docs[n - 1]["text"].split()[:60])}
        for n in sorted(used)
    ]
    trace = [f"citation check -> {len(citations)} valid sources cited"]
    if docs and not citations:
        answer += "\n\n_Note: this answer has no valid inline citations; treat it with caution._"
        trace.append("warning: ungrounded answer")
    return {"answer": answer, "citations": citations, "trace": trace}

@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(AgentState)
    g.add_node("plan_node", plan_node)
    g.add_node("retrieve_node", retrieve_node)
    g.add_node("grade_node", grade_node)
    g.add_node("rewrite_node", rewrite_node)
    g.add_node("answer_node", answer_node)
    g.add_node("cite_node", cite_node)
    g.add_edge(START, "plan_node")
    g.add_edge("plan_node", "retrieve_node")
    g.add_edge("retrieve_node", "grade_node")
    g.add_conditional_edges("grade_node", route_after_grade, {"answer": "answer_node", "rewrite": "rewrite_node"})
    g.add_edge("rewrite_node", "retrieve_node")
    g.add_edge("answer_node", "cite_node")
    g.add_edge("cite_node", END)
    return g.compile()

def run_agent(question: str, history: list[dict] | None = None) -> dict:
    """history: [{'role': 'user'|'assistant', 'content': str}, ...] (the caller owns conversation memory)."""
    state = get_graph().invoke(
        {"question": question, "history": history or [], "attempts": 0, "trace": []},
        config={"run_name": "scholaragent_qa", "tags": ["scholaragent"]},
    )
    return {
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "trace": state.get("trace", []),
        "attempts": state.get("attempts", 0),
        "contexts": [d["text"] for d in state.get("docs", [])]
    }

def compare_with_llm(paper_ids: list[str], aspect: str) -> dict:
    evidence = compare_papers(paper_ids, aspect)
    blocks = []
    for p in evidence["papers"]:
        if "error" in p:
            blocks.append(f"### {p['paper_id']}\n(not in index)")
            continue
        passages = "\n".join(f"- ({x['section']}) {' '.join(x['text'].split()[:120])}" for x in p["passages"])
        blocks.append(f"### {p['title']} (arXiv {p['paper_id']}, {p['year']})\n{passages}")
    comparison = ask_llm(COMPARE_PROMPT.format(aspect=aspect, evidence="\n\n".join(blocks)), temperature=0.1)
    return {"comparison": comparison, "evidence": evidence}