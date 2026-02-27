def optimize_prompt(prompt: str, intent: str) -> str:

    if intent == "coding":
        return f"Provide clean, well-documented code and explain the logic clearly:\n\n{prompt}"

    if intent == "education":
        return f"Provide a structured explanation with definitions, examples, and a summary:\n\n{prompt}"

    if intent == "business":
        return f"Write in a professional and concise tone. Structure clearly:\n\n{prompt}"

    if intent == "creative":
        return f"Enhance creativity, imagery, and engagement:\n\n{prompt}"

    return prompt
