from typing import Tuple


def detect_intent(prompt: str) -> str:

    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["code", "python", "function", "bug", "error"]):
        return "coding"

    if any(word in prompt_lower for word in ["essay", "explain", "definition", "theory"]):
        return "education"

    if any(word in prompt_lower for word in ["email", "proposal", "client", "business"]):
        return "business"

    if any(word in prompt_lower for word in ["story", "creative", "poem"]):
        return "creative"

    return "general"
