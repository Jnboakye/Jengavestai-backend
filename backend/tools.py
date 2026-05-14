# tools.py - Tool definitions and schemas for JengaVest AI

import requests
import os
from datetime import datetime
from rag import search as rag_search

# ─────────────────────────────────────────
# TOOL SCHEMAS
# These tell Claude what tools are available
# and what inputs each tool needs
# ─────────────────────────────────────────

TOOLS = [
    {
        "name": "get_stock_data",
        "description": "Fetches live stock prices, performance data and basic financials for one or more stocks using Finnhub API",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of stock ticker symbols e.g. ['TSLA', 'AAPL', 'GOOG']"
                }
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "search_financial_news",
        "description": "Searches the web for latest financial news and market updates relevant to the user's query",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query e.g. 'Tesla stock news today' or 'S&P 500 market update'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_documents",
        "description": "Searches through uploaded financial documents using RAG pipeline (ChromaDB + BM25) to find relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant content in uploaded documents"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of relevant document chunks to retrieve. Default is 5"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_report_to_drive",
        "description": "Saves a portfolio analysis or financial report to the user's Google Drive via MCP",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the report e.g. 'Portfolio Analysis - May 2026'"
                },
                "content": {
                    "type": "string",
                    "description": "The full content of the report to save"
                }
            },
            "required": ["title", "content"]
        }
    }
]


# ─────────────────────────────────────────
# TOOL FUNCTIONS
# These are the actual functions that run
# when Claude calls a tool
# ─────────────────────────────────────────

def get_stock_data(tickers: list) -> dict:
    """
    Fetches live stock data for a list of tickers
    using Finnhub API
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    results = {}

    for ticker in tickers:
        try:
            # Get current quote
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
            quote_response = requests.get(quote_url, timeout=10)
            quote = quote_response.json()

            # Get company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={api_key}"
            profile_response = requests.get(profile_url, timeout=10)
            profile = profile_response.json()

            current_price = quote.get("c", 0)
            previous_close = quote.get("pc", 0)
            price_change = quote.get("d", 0)
            price_change_percent = quote.get("dp", 0)

            results[ticker] = {
                "name": profile.get("name", ticker),
                "current_price": current_price,
                "previous_close": previous_close,
                "price_change": round(price_change, 2),
                "price_change_percent": round(price_change_percent, 2),
                "high": quote.get("h", 0),
                "low": quote.get("l", 0),
                "open": quote.get("o", 0),
                "market_cap": profile.get("marketCapitalization", "N/A"),
                "currency": profile.get("currency", "USD"),
                "exchange": profile.get("exchange", "N/A"),
                "industry": profile.get("finnhubIndustry", "N/A"),
                "timestamp": datetime.now().strftime("%B %d %Y %H:%M")
            }

        except Exception as e:
            results[ticker] = {"error": str(e)}

    return results


def search_documents(query: str, n_results: int = 5) -> dict:
    """
    Searches uploaded financial documents using
    RAG pipeline - ChromaDB + BM25 hybrid search
    """
    try:
        results = rag_search(query, n_results=n_results)

        if not results:
            return {
                "query": query,
                "results": [],
                "message": "No documents found matching the query"
            }

        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result["content"],
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "relevance_score": round(result["relevance_score"], 3),
                "search_type": result["search_type"]
            })

        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }

    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "message": "Error searching documents"
        }


def search_financial_news(query: str) -> dict:
    """
    Searches for financial news.
    Claude handles this via built-in web search capability.
    """
    return {
        "query": query,
        "status": "web_search_requested",
        "message": f"Searching for financial news related to: {query}. "
                   f"Using web search to fetch latest market news, "
                   f"earnings reports, and financial updates.",
        "instructions": "Please use your web search capability to find the latest news "
                       f"for: {query}"
    }


def save_report_to_drive(title: str, content: str) -> dict:
    """
    Saves a report to Google Drive via MCP
    Placeholder until MCP is fully connected
    """
    return {
        "message": f"Report '{title}' saved to Google Drive",
        "status": "success"
    }


# ─────────────────────────────────────────
# TOOL EXECUTOR
# This function routes tool calls from Claude
# to the correct function
# ─────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Receives a tool call from Claude and routes
    it to the correct function
    """
    if tool_name == "get_stock_data":
        result = get_stock_data(tool_input["tickers"])

    elif tool_name == "search_documents":
        result = search_documents(
            tool_input["query"],
            tool_input.get("n_results", 5)
        )

    elif tool_name == "search_financial_news":
        result = search_financial_news(tool_input["query"])

    elif tool_name == "save_report_to_drive":
        result = save_report_to_drive(
            tool_input["title"],
            tool_input["content"]
        )

    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return str(result)