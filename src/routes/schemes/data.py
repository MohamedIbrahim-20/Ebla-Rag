from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str
    chunk_size: Optional[int] = 100  # Default chunk size of 1MB
    overlap_size: Optional[int] = 20      # Default overlap of 200KB
    do_reset: Optional[bool] = False