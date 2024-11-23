from pydantic import BaseModel
from typing import List, Optional

# Request model for Job
class JobFilterRequest(BaseModel):
    job_id: str
    Country: Optional[str] = None
    location: Optional[str] = None
    Education: Optional[str] = None
    Gender: Optional[str] = None
    age: Optional[int] = None
    
# Request model for Candidate
class CandidateFilterRequest(BaseModel):
    can_id: str