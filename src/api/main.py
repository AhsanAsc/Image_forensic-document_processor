from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from typing import List
import time
import asyncio
from pydantic import BaseModel
from io import BytesIO
import logging

# Import our document processor
from src.core.document_processor import DocumentProcessor, DocumentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing API",
    description="API for document deskewing and boundary detection",
    version="1.0.0"
)

# Initialize processor
processor = DocumentProcessor()

class ProcessingResponse(BaseModel):
    """Response model for document processing."""
    angle: float
    confidence: float
    processing_time: float
    image_base64: str
    message: str

class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    results: List[ProcessingResponse]
    total_processing_time: float
    success_count: int
    error_count: int

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: str

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    try:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to encode processed image")

def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded file bytes to opencv image."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Unable to decode image")
            
        return image
    except Exception as e:
        logger.error(f"Error decoding uploaded image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file format")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/process-document",
            "batch": "/process-batch", 
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "document-processor"
    }

@app.post("/process-document", response_model=ProcessingResponse)
async def process_single_document(file: UploadFile = File(...)):
    """
    Process a single document image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns the processed document with rotation angle and confidence score.
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Received: {file.content_type}"
        )
    
    try:
        # Read file
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Decode image
        image = decode_uploaded_image(file_bytes)
        
        # Process document
        start_time = time.time()
        result = processor.process_document(image)
        total_time = time.time() - start_time
        
        # Check if processing took too long
        if total_time > 0.1:  # 100ms threshold
            logger.warning(f"Processing took {total_time*1000:.1f}ms (> 100ms threshold)")
        
        # Convert processed image to base64
        image_base64 = image_to_base64(result.cropped_image)
        
        return ProcessingResponse(
            angle=round(result.angle, 3),
            confidence=round(result.confidence, 3),
            processing_time=round(result.processing_time * 1000, 1),  # Convert to ms
            image_base64=image_base64,
            message="Document processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing document: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/process-batch", response_model=BatchProcessingResponse)
async def process_batch_documents(files: List[UploadFile] = File(...)):
    """
    Process multiple document images in batch.
    
    - **files**: List of image files
    
    Returns results for all processed documents.
    """
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    success_count = 0
    error_count = 0
    batch_start_time = time.time()
    
    for i, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                logger.warning(f"Skipping non-image file {i}: {file.content_type}")
                error_count += 1
                continue
            
            # Read and decode file
            file_bytes = await file.read()
            image = decode_uploaded_image(file_bytes)
            
            # Process document
            result = processor.process_document(image)
            
            # Convert to base64
            image_base64 = image_to_base64(result.cropped_image)
            
            results.append(ProcessingResponse(
                angle=round(result.angle, 3),
                confidence=round(result.confidence, 3),
                processing_time=round(result.processing_time * 1000, 1),
                image_base64=image_base64,
                message=f"Document {i+1} processed successfully"
            ))
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing file {i}: {str(e)}")
            error_count += 1
            # Continue with other files instead of failing entire batch
    
    total_processing_time = round((time.time() - batch_start_time) * 1000, 1)
    
    return BatchProcessingResponse(
        results=results,
        total_processing_time=total_processing_time,
        success_count=success_count,
        error_count=error_count
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            details=f"Error occurred while processing request to {request.url}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details="An unexpected error occurred"
        ).dict()
    )

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests for monitoring."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time*1000:.1f}ms"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )