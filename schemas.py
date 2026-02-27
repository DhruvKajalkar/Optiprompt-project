from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class PromptRequest(BaseModel):
    prompt: str
    prune_ratio: Optional[float] = 0.3


class PromptResponse(BaseModel):
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    intent_type: Optional[str]


class StatsResponse(BaseModel):
    total_prompts: int
    total_tokens_saved: int
    average_compression: float
