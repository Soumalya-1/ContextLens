import concurrent.futures
from ddgs import DDGS

def _run_search(query: str, max_results: int) -> str:
    """
    Core search execution helper to be run in a thread.
    """
    with DDGS() as ddgs:
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


def search_web(query: str, max_results: int = 3) -> str:
    """
    Performs a web search using DuckDuckGo with a 10-second timeout.
    Returns formatted results or an error string on timeout or failure.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_search, query, max_results)
        try:
            return future.result(timeout=10.0)
        except concurrent.futures.TimeoutError:
            print("\n[Web Search] Error: Search timed out after 10 seconds.")
            return "Error: Web search timed out."
        except Exception as e:
            print(f"\n[Web Search] Warning: Search failed due to error: {e}")
            return ""
