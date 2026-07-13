from ddgs import DDGS

def search_web(query: str, max_results: int = 3) -> str:
    """
    Performs a web search using DuckDuckGo and formats the results.
    Returns a concatenated string of titles, URLs, and snippets.
    Returns an empty string if search fails or finds no results.
    """
    try:
        with DDGS() as ddgs:
            # text query returns results containing title, href, and body
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return ""
            
            formatted_results = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                href = r.get("href", "")
                body = r.get("body", "")
                formatted_results.append(
                    f"Result {i}:\n"
                    f"Title: {title}\n"
                    f"URL: {href}\n"
                    f"Content: {body}\n"
                )
            return "\n".join(formatted_results)
    except Exception as e:
        # Gracefully handle search exceptions (e.g., rate limits, network issues)
        print(f"\n[Web Search] Warning: Search failed due to error: {e}")
        return ""
