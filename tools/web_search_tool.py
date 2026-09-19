"""
Web search tool, backed by Tavily's API — built specifically for LLM
agent consumption (structured, relevance-ranked results), rather than
scraping a search engine's HTML.
"""

import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(query: str) -> str:
    """
    Search the web for `query` using Tavily and return a short text
    summary of the top results.

    Args:
        query: The search query string.

    Returns:
        A plain-text summary of the top results, or an error message
        if the search fails.
    """
    try:
        response = _client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return f"No results found for '{query}'."
        summary = "\n".join(
            f"- {r['title']}: {r['content'][:200]}" for r in results
        )
        return summary
    except Exception as exc:
        return f"[ERROR] Web search failed: {exc}"