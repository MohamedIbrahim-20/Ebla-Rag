from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str
    chunk_size: Optional[int] = 100  # Default chunk size of 1MB
    overlap_size: Optional[int] = 20      # Default overlap of 200KB
    do_reset: Optional[bool] = False

class IndexRequest(BaseModel):
    csv_file: Optional[str] = "tech_100_long_real.csv"

class SearchRequest(BaseModel):
    query: str
    method: Optional[str] = "langchain"  # "langchain" or "llamaindex"
    top_k: Optional[int] = 5

class AskRequest(BaseModel):
    question: str
    method: Optional[str] = "langchain"  # "langchain" or "llamaindex"