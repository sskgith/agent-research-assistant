"""
Web search tool (stub).

Real implementation will call a search API (e.g. Tavily, SerpAPI, or a
plain requests-based search). For now this returns a hardcoded result so
the graph plumbing can be tested without any API keys.
"""


def web_search(query: str) -> str:
    """
    Search the web for `query` and return a short text summary of results.

    Args:
        query: The search query string.

    Returns:
        A plain-text summary of the top result(s).
    """
    # TODO: replace with a real search API call.
    return f"[STUB RESULT] Top result for '{query}': placeholder summary text."
