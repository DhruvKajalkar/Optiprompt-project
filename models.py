from sqlalchemy import Column, Integer, Text, Float, DateTime
from sqlalchemy.sql import func
from database import Base


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text)
    compressed_text = Column(Text)
    intent_type = Column(Text)
    original_tokens = Column(Integer)
    compressed_tokens = Column(Integer)
    compression_ratio = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
