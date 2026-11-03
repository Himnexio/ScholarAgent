from __future__ import annotations
import logging
import sys
from mcp.server.fastmcp import FastMCP
from scholaragent import tools

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
mcp = FastMCP("scholaragent")

@mcp.tool()
def search_papers(query: str, k: int = 5) -> list[dict]:
    """Search the indexed arXiv papers and return the k most relevant papers with a best-matching snippet."""
    return tools.search_papers(query, k=min(max(k, 1), 10))

@mcp.tool()
def get_paper(paper_id: str) -> dict:
    """Get title, authors, year, abstract and URL of an indexed paper by its arXiv id (e.g. '2401.12345')."""
    return tools.get_paper(paper_id) or {"error": f"{paper_id} is not in the index"}

@mcp.tool()
def get_paper_evidence(paper_id: str, question: str, k: int = 3) -> list[dict]:
    """Return the k passages of ONE paper that best match a question (e.g. 'which datasets were used?')."""
    return tools.get_paper_evidence(paper_id, question, k=min(max(k, 1), 8))

@mcp.tool()
def compare_papers(paper_ids: list[str], aspect: str, k_per_paper: int = 3) -> dict:
    """Collect passages about the same aspect (e.g. 'training data', 'limitations') from 2-4 papers to compare."""
    return tools.compare_papers(paper_ids[:4], aspect, k_per_paper=min(max(k_per_paper, 1), 5))

def main() -> None:
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)

if __name__ == "__main__":
    main()