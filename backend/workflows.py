# workflows.py - Query routing, chaining, and parallelization workflows

import asyncio
import re
from prompts import (
    MAIN_SYSTEM_PROMPT,
    PORTFOLIO_ANALYSIS_PROMPT,
    MARKET_NEWS_PROMPT,
    DOCUMENT_ANALYSIS_PROMPT,
    REBALANCING_PROMPT
)
from tools import get_stock_data, search_documents

# ─────────────────────────────────────────────────────────
# 1. ROUTING WORKFLOW
# ─────────────────────────────────────────────────────────

def route_query(user_message: str) -> dict:
    """
    Routes user query to appropriate workflow and selects
    the relevant prompt based on query intent.
    
    Args:
        user_message: User's input message
    
    Returns:
        Dict with:
        - query_type: str (portfolio|news|document|rebalancing|general)
        - prompt: str (the appropriate system prompt)
        - reason: str (brief explanation of routing decision)
    """
    
    # Convert to lowercase for keyword matching
    message_lower = user_message.lower()
    
    # Portfolio queries
    portfolio_keywords = [
        "portfolio", "holdings", "allocation", "diversification",
        "my stocks", "my investments", "what should i own", "asset allocation"
    ]
    if any(keyword in message_lower for keyword in portfolio_keywords):
        return {
            "query_type": "portfolio",
            "prompt": PORTFOLIO_ANALYSIS_PROMPT,
            "reason": "User asking about portfolio analysis"
        }
    
    # News queries
    news_keywords = [
        "news", "market update", "breaking", "latest", "trending",
        "what's happening", "market conditions", "today's market",
        "stock news", "earnings report"
    ]
    if any(keyword in message_lower for keyword in news_keywords):
        return {
            "query_type": "news",
            "prompt": MARKET_NEWS_PROMPT,
            "reason": "User asking about market news"
        }
    
    # Document queries
    document_keywords = [
        "document", "report", "filing", "earnings", "pdf", "analysis",
        "uploaded", "10-k", "annual report", "quarterly", "sec filing"
    ]
    if any(keyword in message_lower for keyword in document_keywords):
        return {
            "query_type": "document",
            "prompt": DOCUMENT_ANALYSIS_PROMPT,
            "reason": "User asking about uploaded financial documents"
        }
    
    # Rebalancing queries
    rebalancing_keywords = [
        "rebalance", "rebalancing", "should i sell", "should i buy",
        "adjust portfolio", "shift allocation", "reduce exposure",
        "increase exposure", "portfolio optimization"
    ]
    if any(keyword in message_lower for keyword in rebalancing_keywords):
        return {
            "query_type": "rebalancing",
            "prompt": REBALANCING_PROMPT,
            "reason": "User asking about portfolio rebalancing"
        }
    
    # Default to general financial question
    return {
        "query_type": "general",
        "prompt": MAIN_SYSTEM_PROMPT,
        "reason": "General financial question"
    }


# ─────────────────────────────────────────────────────────
# 2. CHAINING WORKFLOW
# ─────────────────────────────────────────────────────────

def extract_tickers_from_message(message: str) -> list:
    """
    Extracts stock ticker symbols from a message.
    Looks for patterns like $AAPL, AAPL, or text like "Apple"
    
    Args:
        message: User message
    
    Returns:
        List of potential ticker symbols
    """
    # Look for $TICKER or TICKER patterns (uppercase, 1-5 chars)
    ticker_pattern = r'\$?([A-Z]{1,5})(?:\s|$|,)'
    matches = re.findall(ticker_pattern, message)
    
    # Common company keywords to ticker mappings
    keyword_to_ticker = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "meta": "META",
        "facebook": "META",
        "s&p 500": "^GSPC",
        "nasdaq": "^IXIC",
    }
    
    tickers = list(set(matches))  # Remove duplicates
    
    # Add keyword matches
    for keyword, ticker in keyword_to_ticker.items():
        if keyword in message.lower():
            if ticker not in tickers:
                tickers.append(ticker)
    
    return tickers[:10]  # Limit to 10 tickers


def chain_analysis(user_message: str) -> dict:
    """
    Chains multiple tool calls together in sequence.
    Each step passes output to next step.
    
    Step 1: Extract tickers and fetch stock data
    Step 2: Search for relevant documents
    Step 3: Synthesize findings
    
    Args:
        user_message: User's query
    
    Returns:
        Dict with combined analysis context
    """
    analysis_context = {
        "user_message": user_message,
        "steps_completed": [],
        "stock_data": {},
        "documents": [],
        "combined_context": ""
    }
    
    try:
        # STEP 1: Extract tickers and fetch stock data
        print("Chaining Step 1: Fetching stock data...")
        tickers = extract_tickers_from_message(user_message)
        
        if tickers:
            stock_data = get_stock_data(tickers)
            analysis_context["stock_data"] = stock_data
            analysis_context["steps_completed"].append("stock_data_fetched")
        
        # STEP 2: Search for relevant documents
        print("Chaining Step 2: Searching documents...")
        doc_results = search_documents(user_message, n_results=5)
        
        if doc_results.get("results"):
            analysis_context["documents"] = doc_results["results"]
            analysis_context["steps_completed"].append("documents_searched")
        
        # STEP 3: Synthesize findings
        print("Chaining Step 3: Synthesizing analysis...")
        context_parts = []
        
        if analysis_context["stock_data"]:
            context_parts.append(f"Stock Data:\n{analysis_context['stock_data']}")
        
        if analysis_context["documents"]:
            docs_summary = "\n".join([
                f"- {doc['document_name']} (Page {doc['page_number']}): {doc['content'][:200]}..."
                for doc in analysis_context["documents"][:3]
            ])
            context_parts.append(f"Relevant Documents:\n{docs_summary}")
        
        analysis_context["combined_context"] = "\n\n".join(context_parts)
        analysis_context["steps_completed"].append("analysis_synthesized")
        
    except Exception as e:
        analysis_context["error"] = str(e)
        print(f"Error in chain_analysis: {e}")
    
    return analysis_context


# ─────────────────────────────────────────────────────────
# 3. PARALLELIZATION WORKFLOW
# ─────────────────────────────────────────────────────────

async def fetch_stock_data_async(tickers: list) -> dict:
    """Async wrapper for stock data fetching"""
    return await asyncio.to_thread(get_stock_data, tickers)


async def search_documents_async(query: str) -> dict:
    """Async wrapper for document search"""
    return await asyncio.to_thread(search_documents, query, 5)


async def parallel_fetch(user_message: str) -> dict:
    """
    Simultaneously fetches data from multiple sources using asyncio.
    Combines results in a single context.
    
    Args:
        user_message: User's query
    
    Returns:
        Dict with combined parallel fetch results
    """
    parallel_context = {
        "user_message": user_message,
        "stock_data": {},
        "documents": [],
        "combined_context": "",
        "fetch_time_ms": 0
    }
    
    try:
        import time
        start_time = time.time()
        
        # Extract tickers from message
        tickers = extract_tickers_from_message(user_message)
        
        # Create async tasks
        tasks = []
        
        # Task 1: Fetch stock data
        if tickers:
            tasks.append(fetch_stock_data_async(tickers))
        else:
            tasks.append(asyncio.sleep(0))  # Placeholder if no tickers
        
        # Task 2: Search documents
        tasks.append(search_documents_async(user_message))
        
        # Execute tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        if len(results) > 0 and isinstance(results[0], dict):
            parallel_context["stock_data"] = results[0]
        
        if len(results) > 1 and isinstance(results[1], dict) and results[1].get("results"):
            parallel_context["documents"] = results[1]["results"]
        
        # Combine context
        context_parts = []
        
        if parallel_context["stock_data"]:
            context_parts.append(f"Stock Data:\n{parallel_context['stock_data']}")
        
        if parallel_context["documents"]:
            docs_summary = "\n".join([
                f"- {doc['document_name']} (Page {doc['page_number']}): {doc['content'][:200]}..."
                for doc in parallel_context["documents"][:3]
            ])
            context_parts.append(f"Relevant Documents:\n{docs_summary}")
        
        parallel_context["combined_context"] = "\n\n".join(context_parts)
        parallel_context["fetch_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
    except Exception as e:
        parallel_context["error"] = str(e)
        print(f"Error in parallel_fetch: {e}")
    
    return parallel_context


def run_parallel_fetch(user_message: str) -> dict:
    """
    Synchronous wrapper for parallel_fetch.
    Runs async parallel_fetch in event loop.
    
    Args:
        user_message: User's query
    
    Returns:
        Dict with parallel fetch results
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(parallel_fetch(user_message))
