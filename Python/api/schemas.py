from pydantic import BaseModel
from typing import List, Optional

class InputData(BaseModel):
    user_id: int
    features: List[float]
    model_type: str  # "arrhythmia" or "binary"
    device_info: Optional[str] = "Unknown-Device"

class FeedbackData(BaseModel):
    prediction_id: int
    user_id: int
    is_correct: bool
    corrected_label: Optional[str] = None
    rating: Optional[int] = None
    comments: Optional[str] = None
