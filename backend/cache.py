# cache.py - Prompt caching utility for Anthropic API

from prompts import MAIN_SYSTEM_PROMPT

# ─────────────────────────────────────────────────────────
# PROMPT CACHING WITH ANTHROPIC
# ─────────────────────────────────────────────────────────

def build_cached_prompt():
    """
    Builds the system prompt with cache_control wrapper.
    This enables Anthropic's prompt caching feature which:
    - Caches the system prompt on first use
    - Reduces latency and cost on subsequent uses
    - Uses ephemeral cache (valid for 5 minutes)
    
    Returns a dict ready to pass to Anthropic API
    """
    return {
        "type": "text",
        "text": MAIN_SYSTEM_PROMPT,
        "cache_control": {
            "type": "ephemeral"
        }
    }


def build_messages(conversation_history: list, user_message: str) -> list:
    """
    Builds the messages array for Anthropic API.
    
    Args:
        conversation_history: List of previous messages in format:
            [{"role": "user/assistant", "content": "..."}, ...]
        user_message: The current user message
    
    Returns:
        List of messages in Anthropic API format, including 
        conversation history and new user message
    """
    # Start with conversation history if provided
    messages = conversation_history.copy() if conversation_history else []
    
    # Add the new user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages
