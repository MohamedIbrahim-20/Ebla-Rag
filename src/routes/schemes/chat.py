from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    method: Optional[str] = "langchain"  # or "llamaindex"
    temperature: Optional[float] = None
    create_only: Optional[bool] = False


