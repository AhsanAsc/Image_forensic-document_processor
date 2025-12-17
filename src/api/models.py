from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    max_dimension: int = Field(default=1024, description="Maximum image dimension for processing")
    padding: int = Field(default=10, description="Padding to add around cropped document")
    angle_threshold: float = Field(default=0.5, description="Minimum angle for rotation correction")
    
class DocumentStats(BaseModel):
    """Statistics about the processed document."""
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    rotation_angle: float
    confidence_score: float

class DetailedResponse(ProcessingResponse):
    """Extended response with additional details."""
    stats: DocumentStats
    processing_steps: List[str]
    warnings: Optional[List[str]] = None