from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import engine, SessionLocal
from models import Base, Prompt
from schemas import PromptRequest, PromptResponse, StatsResponse
from entropy_engine import compress_prompt

Base.metadata.create_all(bind=engine)

app = FastAPI(title="OptiPrompt Backend")


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/compress", response_model=PromptResponse)
def compress(request: PromptRequest, db: Session = Depends(get_db)):

    compressed_text, stats = compress_prompt(
        request.prompt,
        request.prune_ratio
    )

    # Save to database
    prompt_entry = Prompt(
        original_text=request.prompt,
        compressed_text=compressed_text,
        intent_type="unknown",
        original_tokens=stats["original_tokens"],
        compressed_tokens=stats["compressed_tokens"],
        compression_ratio=stats["compression_ratio"]
    )

    db.add(prompt_entry)
    db.commit()
    db.refresh(prompt_entry)

    return PromptResponse(
        original_prompt=request.prompt,
        compressed_prompt=compressed_text,
        original_tokens=stats["original_tokens"],
        compressed_tokens=stats["compressed_tokens"],
        compression_ratio=stats["compression_ratio"],
        intent_type="unknown"
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):

    prompts = db.query(Prompt).all()

    total_prompts = len(prompts)
    total_tokens_saved = sum(
        p.original_tokens - p.compressed_tokens for p in prompts
    )

    average_compression = (
        sum(p.compression_ratio for p in prompts) / total_prompts
        if total_prompts > 0 else 0
    )

    return StatsResponse(
        total_prompts=total_prompts,
        total_tokens_saved=total_tokens_saved,
        average_compression=round(average_compression, 3)
    )
