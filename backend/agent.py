# agent.py - Main JengaVest AI agent with tool use and streaming

import os
import json
from typing import Generator
from urllib.parse import urlparse
from dotenv import load_dotenv
from anthropic import Anthropic

from prompts import MAIN_SYSTEM_PROMPT
from tools import TOOLS, execute_tool
from cache import build_cached_prompt, build_messages
from workflows import route_query, run_parallel_fetch

# Load environment variables
load_dotenv()

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search"
}


# ─────────────────────────────────────────────────────────
# CITATION EXTRACTION
#
# Sources come from what the model actually used, not from parsing its
# prose. The prompts forbid inline brackets, so scraping the text for
# "[...]" returned nothing. Instead we read them off the web_search
# result blocks and the RAG tool results.
# ─────────────────────────────────────────────────────────

def _source_label(url: str, title: str) -> str:
    """Turns a result into a short pill label, preferring the domain."""
    try:
        netloc = urlparse(url).netloc if url else ""
        if netloc.startswith("www."):
            netloc = netloc[4:]
    except Exception:
        netloc = ""

    label = netloc or (title or "")
    return label[:60]


def collect_citations_from_response(response, citations: list) -> None:
    """
    Appends source labels found in a Claude response to `citations`.

    Server-side web search returns its results inline as
    web_search_tool_result blocks, so every response in the agent loop
    needs to be scanned, not just the final one.
    """
    try:
        for block in response.content:
            block_type = getattr(block, "type", None)

            if block_type == "web_search_tool_result":
                results = getattr(block, "content", None) or []
                if not isinstance(results, list):
                    continue  # an error object rather than results
                for item in results:
                    label = _source_label(
                        getattr(item, "url", "") or "",
                        getattr(item, "title", "") or ""
                    )
                    if label:
                        citations.append(label)

            elif block_type == "text":
                for citation in getattr(block, "citations", None) or []:
                    label = _source_label(
                        getattr(citation, "url", "") or "",
                        getattr(citation, "title", "") or ""
                    )
                    if label:
                        citations.append(label)

    except Exception as e:
        print(f"Error collecting citations: {e}")


def dedupe_preserving_order(items: list) -> list:
    """Removes duplicates while keeping first-seen order."""
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

# ─────────────────────────────────────────────────────────
# AGENT INITIALIZATION
# ─────────────────────────────────────────────────────────

class JengaVestAgent:
    """Main AI agent for financial analysis"""
    
    def __init__(self):
        """Initialize the agent and perform environment checks"""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self.status = {}
        
        # Initialize client
        self._initialize_client()
        
        # Run environment checks
        self._check_environment()
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        if not self.api_key:
            print("❌ ANTHROPIC_API_KEY not found in .env")
            self.status["anthropic_client"] = "FAILED"
            return
        
        try:
            self.client = Anthropic(api_key=self.api_key)
            self.status["anthropic_client"] = "OK"
            print("✓ Anthropic client initialized")
        except Exception as e:
            print(f"❌ Error initializing Anthropic client: {e}")
            self.status["anthropic_client"] = f"FAILED: {e}"
    
    def _check_environment(self):
        """Check all tools and dependencies are available"""
        print("\n" + "="*50)
        print("JENGAVEST AI - ENVIRONMENT CHECK")
        print("="*50)
        
        # Check Anthropic client
        if self.status.get("anthropic_client") == "OK":
            print("✓ Anthropic API configured")
        else:
            print("❌ Anthropic API not configured")
        
        # Check tools
        print(f"✓ {len(TOOLS)} tools registered:")
        for tool in TOOLS:
            print(f"  - {tool['name']}")
        
        # Check environment variables
        print("\nEnvironment Variables:")
        if self.api_key:
            print(f"✓ ANTHROPIC_API_KEY loaded")
        else:
            print(f"❌ ANTHROPIC_API_KEY not set")
        
        # Check imports
        print("\nDependencies:")
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        if finnhub_key:
            print("✓ Finnhub API key loaded")
            self.status["yfinance"] = "OK"
        else:
            print("❌ FINNHUB_API_KEY not set")
            self.status["yfinance"] = "FAILED"
        
        try:
            import chromadb
            print("✓ ChromaDB available")
            self.status["chromadb"] = "OK"
        except ImportError:
            print("❌ ChromaDB not available")
            self.status["chromadb"] = "FAILED"
        
        try:
            import rank_bm25
            print("✓ rank_bm25 available")
            self.status["rank_bm25"] = "OK"
        except ImportError:
            print("❌ rank_bm25 not available")
            self.status["rank_bm25"] = "FAILED"
        
        print("\n" + "="*50 + "\n")
    
    # ─────────────────────────────────────────────────────────
    # AGENT LOOP WITH TOOL USE
    # ─────────────────────────────────────────────────────────
    
    def run_agent(self, user_message: str, conversation_history: list = None) -> dict:
        """
        Main agent loop that handles tool use and returns final response.
        
        Args:
            user_message: User's input message
            conversation_history: List of previous messages
        
        Returns:
            Dict with:
            - response: str (final AI response)
            - citations: list (sources cited)
            - tools_used: list (tools called)
            - route: str (query type)
        """
        
        if not self.client:
            return {
                "response": "Error: Anthropic API not configured",
                "citations": [],
                "tools_used": [],
                "route": "error"
            }
        
        # Initialize tracking
        conversation_history = conversation_history or []
        tools_used = []
        citations = []
        
        # Route query and get appropriate prompt
        route_result = route_query(user_message)
        query_type = route_result["query_type"]
        task_prompt = route_result["prompt"]
        
        print(f"Query routed as: {query_type}")
        
        # Build messages for API
        messages = build_messages(conversation_history, user_message)
        
        # Build system prompt with caching
        system_prompt = build_cached_prompt()
        
        # First API call with extended thinking
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=16000,
                temperature=1,  # Required for extended thinking
                thinking={
                    "type": "enabled",
                    "budget_tokens": 8000
                },
                system=[
                    system_prompt,
                    {
                        "type": "text",
                        "text": task_prompt
                    }
                ],
                tools=TOOLS + [WEB_SEARCH_TOOL],
                messages=messages
            )
        except Exception as e:
            return {
                "response": f"Error calling Claude API: {str(e)}",
                "citations": [],
                "tools_used": [],
                "route": query_type
            }

        collect_citations_from_response(response, citations)

        # Handle agentic loop with tool use
        while response.stop_reason == "tool_use":
            # Find tool_use blocks
            tool_use_blocks = [block for block in response.content 
                              if block.type == "tool_use"]
            
            # Process each tool call
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tool_use_id = tool_block.id
                
                print(f"Tool called: {tool_name}")
                tools_used.append(tool_name)
                
                # Execute tool (citations collects document sources)
                try:
                    result = execute_tool(tool_name, tool_input, citations)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": str(result)
                })
            
            # Add assistant response and tool results to messages
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            messages.append({
                "role": "user",
                "content": tool_results
            })
            
            # Continue conversation with tool results
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=16000,
                    temperature=1,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 8000
                    },
                    system=[
                        system_prompt,
                        {
                            "type": "text",
                            "text": task_prompt
                        }
                    ],
                    tools=TOOLS + [WEB_SEARCH_TOOL],
                    messages=messages
                )
            except Exception as e:
                return {
                    "response": f"Error in agent loop: {str(e)}",
                    "citations": dedupe_preserving_order(citations),
                    "tools_used": tools_used,
                    "route": query_type
                }

            collect_citations_from_response(response, citations)

        # Extract final text response
        final_response = ""
        for block in response.content:
            if block.type == "text":
                final_response += block.text

        return {
            "response": final_response,
            "citations": dedupe_preserving_order(citations),
            "tools_used": dedupe_preserving_order(tools_used),
            "route": query_type
        }
    
    # ─────────────────────────────────────────────────────────
    # STREAMING RESPONSE
    # ─────────────────────────────────────────────────────────
    
    def stream_agent(self, user_message: str, conversation_history: list = None) -> Generator:
        """
        Streams agent response in real-time using SSE format.
        
        Args:
            user_message: User's input message
            conversation_history: List of previous messages
        
        Yields:
            SSE formatted chunks: "data: {json}\n\n"
        """
        
        if not self.client:
            yield f"data: {json.dumps({'error': 'API not configured'})}\n\n"
            return
        
        conversation_history = conversation_history or []
        tools_used = []
        
        # Route query
        route_result = route_query(user_message)
        query_type = route_result["query_type"]
        task_prompt = route_result["prompt"]
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'Routed as: {query_type}'})}\n\n"
        
        # Build messages
        messages = build_messages(conversation_history, user_message)
        system_prompt = build_cached_prompt()
        
        # First API call with streaming
        try:
            with self.client.messages.stream(
                model="claude-sonnet-4-5",
                max_tokens=16000,
                temperature=1,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 8000
                },
                system=[
                    system_prompt,
                    {
                        "type": "text",
                        "text": task_prompt
                    }
                ],
                tools=TOOLS + [WEB_SEARCH_TOOL],
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'delta', 'text': text})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ─────────────────────────────────────────────────────────
# SINGLETON AGENT INSTANCE
# ─────────────────────────────────────────────────────────

_agent_instance = None

def get_agent() -> JengaVestAgent:
    """Get or create singleton agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = JengaVestAgent()
    return _agent_instance
