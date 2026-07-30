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

# Fast model for snappy, streaming responses. Extended thinking is disabled so
# text starts streaming immediately rather than after a long thinking phase.
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4000
TEMPERATURE = 0.7

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

        self._initialize_client()
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
        print("\n" + "=" * 50)
        print("JENGAVEST AI - ENVIRONMENT CHECK")
        print("=" * 50)

        if self.status.get("anthropic_client") == "OK":
            print(f"✓ Anthropic API configured (model: {MODEL})")
        else:
            print("❌ Anthropic API not configured")

        print(f"✓ {len(TOOLS)} tools registered:")
        for tool in TOOLS:
            print(f"  - {tool['name']}")

        print("\nEnvironment Variables:")
        print("✓ ANTHROPIC_API_KEY loaded" if self.api_key else "❌ ANTHROPIC_API_KEY not set")

        finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.status["finnhub"] = "OK" if finnhub_key else "FAILED"
        print("✓ Finnhub API key loaded" if finnhub_key else "❌ FINNHUB_API_KEY not set")

        try:
            import chromadb  # noqa: F401
            self.status["chromadb"] = "OK"
            print("✓ ChromaDB available")
        except ImportError:
            self.status["chromadb"] = "FAILED"
            print("❌ ChromaDB not available")

        print("\n" + "=" * 50 + "\n")

    # ─────────────────────────────────────────────────────────
    # SHARED REQUEST HELPERS
    # ─────────────────────────────────────────────────────────

    def _system(self, task_prompt: str):
        return [build_cached_prompt(), {"type": "text", "text": task_prompt}]

    # ─────────────────────────────────────────────────────────
    # AGENT LOOP WITH TOOL USE (non-streaming — used as a fallback)
    # ─────────────────────────────────────────────────────────

    def run_agent(self, user_message: str, conversation_history: list = None) -> dict:
        """Main agent loop that handles tool use and returns the final response."""
        if not self.client:
            return {"response": "Error: Anthropic API not configured", "citations": [], "tools_used": [], "route": "error"}

        conversation_history = conversation_history or []
        tools_used = []
        citations = []

        route_result = route_query(user_message)
        query_type = route_result["query_type"]
        task_prompt = route_result["prompt"]
        print(f"Query routed as: {query_type}")

        messages = build_messages(conversation_history, user_message)
        system = self._system(task_prompt)

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system,
                tools=TOOLS + [WEB_SEARCH_TOOL],
                messages=messages,
            )
        except Exception as e:
            return {"response": f"Error calling Claude API: {str(e)}", "citations": [], "tools_used": [], "route": query_type}

        collect_citations_from_response(response, citations)

        while response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []
            for tool_block in tool_use_blocks:
                tools_used.append(tool_block.name)
                try:
                    result = execute_tool(tool_block.name, tool_block.input, citations)
                except Exception as e:
                    result = f"Error executing {tool_block.name}: {str(e)}"
                tool_results.append({"type": "tool_result", "tool_use_id": tool_block.id, "content": str(result)})

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=system,
                    tools=TOOLS + [WEB_SEARCH_TOOL],
                    messages=messages,
                )
            except Exception as e:
                return {"response": f"Error in agent loop: {str(e)}", "citations": dedupe_preserving_order(citations), "tools_used": tools_used, "route": query_type}

            collect_citations_from_response(response, citations)

        final_response = "".join(b.text for b in response.content if b.type == "text")
        return {
            "response": final_response,
            "citations": dedupe_preserving_order(citations),
            "tools_used": dedupe_preserving_order(tools_used),
            "route": query_type,
        }

    # ─────────────────────────────────────────────────────────
    # STREAMING RESPONSE (with tool use)
    # ─────────────────────────────────────────────────────────

    def stream_agent(self, user_message: str, conversation_history: list = None) -> Generator:
        """
        Streams the agent response as SSE. Handles the full tool-use loop:
        text streams as the model writes, tools are executed between turns,
        and the final answer streams to the client. Emits:
          {"type":"delta","text": "..."}   incremental text
          {"type":"sources","citations":[...]}
          {"type":"done"}
          {"type":"error","message": "..."}
        """
        if not self.client:
            yield f"data: {json.dumps({'type': 'error', 'message': 'API not configured'})}\n\n"
            return

        conversation_history = conversation_history or []
        citations = []

        route_result = route_query(user_message)
        task_prompt = route_result["prompt"]
        messages = build_messages(conversation_history, user_message)
        system = self._system(task_prompt)

        try:
            while True:
                with self.client.messages.stream(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=system,
                    tools=TOOLS + [WEB_SEARCH_TOOL],
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        yield f"data: {json.dumps({'type': 'delta', 'text': text})}\n\n"
                    final = stream.get_final_message()

                collect_citations_from_response(final, citations)

                if final.stop_reason == "tool_use":
                    tool_blocks = [b for b in final.content if b.type == "tool_use"]
                    tool_results = []
                    for tb in tool_blocks:
                        try:
                            result = execute_tool(tb.name, tb.input, citations)
                        except Exception as e:
                            result = f"Error executing {tb.name}: {str(e)}"
                        tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": str(result)})

                    messages.append({"role": "assistant", "content": final.content})
                    messages.append({"role": "user", "content": tool_results})
                    continue

                break

            if citations:
                yield f"data: {json.dumps({'type': 'sources', 'citations': dedupe_preserving_order(citations)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

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
