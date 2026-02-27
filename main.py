from fastapi import FastAPI
from pydantic import BaseModel
from entropy_engine import compress_prompt

from database import engine
from models import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Entropy Prompt Optimizer API",
    description="Shannon Surprisal-based prompt compression backend",
    version="1.0"
)


class PromptRequest(BaseModel):
    prompt: str
    prune_ratio: float = 0.3


@app.get("/")
def root():
    return {"message": "Entropy Prompt Backend Running 🚀"}


@app.post("/compress")
def compress(request: PromptRequest):
    compressed_text, stats = compress_prompt(
        request.prompt,
        request.prune_ratio
    )

    return {
        "original_prompt": request.prompt,
        "compressed_prompt": compressed_text,
        "stats": stats
    }
