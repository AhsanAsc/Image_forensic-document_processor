import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io
import numpy as np
import cv2
import base64

from src.api.main import app

class TestDocumentProcessingAPI:
    """Test suite for FastAPI endpoints."""
    
    def setup_method(self):
        """Setup test client and fixtures."""
        self.client = TestClient(app)
        
        # Create test image data
        self.test_image = self.create_test_image()
        self.test_image_bytes = self.image_to_bytes(self.test_image)
    
    def create_test_image(self):
        """Create a simple test image."""
        image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (350, 250), (0, 0, 0), 2)
        return image
    
    def image_to_bytes(self, image):
        """Convert numpy image to bytes."""
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert data["endpoints"]["process"] == "/process-document"
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "document-processor"
    
    def test_process_single_document_success(self):
        """Test successful single document processing."""
        files = {"file": ("test.jpg", self.test_image_bytes, "image/jpeg")}
        
        response = self.client.post("/process-document", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "angle" in data
        assert "confidence" in data
        assert "processing_time" in data
        assert "image_base64" in data
        assert "message" in data
        
        # Check data types and ranges
        assert isinstance(data["angle"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["processing_time"] > 0
        assert isinstance(data["image_base64"], str)
        assert len(data["image_base64"]) > 0
    
    def test_process_document_invalid_file_type(self):
        """Test processing with invalid file type."""
        text_content = b"This is not an image"
        files = {"file": ("test.txt", text_content, "text/plain")}
        
        response = self.client.post("/process-document", files=files)
        
        assert response.status_code == 400
        assert "must be an image" in response.json()["error"].lower()
    
    def test_process_document_empty_file(self):
        """Test processing with empty file."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        
        response = self.client.post("/process-document", files=files)
        
        assert response.status_code == 400
        assert "empty file" in response.json()["error"].lower()
    
    def test_process_document_corrupted_image(self):
        """Test processing with corrupted image data."""
        corrupted_data = b"Not a valid image file content"
        files = {"file": ("corrupted.jpg", corrupted_data, "image/jpeg")}
        
        response = self.client.post("/process-document", files=files)
        
        assert response.status_code == 400
        assert "invalid image" in response.json()["error"].lower()
    
    def test_batch_processing_success(self):
        """Test successful batch document processing."""
        files = [
            ("files", ("test1.jpg", self.test_image_bytes, "image/jpeg")),
            ("files", ("test2.jpg", self.test_image_bytes, "image/jpeg"))
        ]
        
        response = self.client.post("/process-batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "results" in data
        assert "total_processing_time" in data
        assert "success_count" in data
        assert "error_count" in data
        
        # Check results
        assert len(data["results"]) == 2
        assert data["success_count"] == 2
        assert data["error_count"] == 0
        
        # Check individual results
        for result in data["results"]:
            assert "angle" in result
            assert "confidence" in result
            assert "image_base64" in result
    
    def test_batch_processing_empty_request(self):
        """Test batch processing with no files."""
        response = self.client.post("/process-batch", files=[])
        
        assert response.status_code == 400
        assert "no files provided" in response.json()["error"].lower()
    
    def test_batch_processing_too_many_files(self):
        """Test batch processing with too many files."""
        # Create more than 10 files
        files = [(f"files", (f"test{i}.jpg", self.test_image_bytes, "image/jpeg")) 
                for i in range(15)]
        
        response = self.client.post("/process-batch", files=files)
        
        assert response.status_code == 400
        assert "maximum 10 files" in response.json()["error"].lower()
    
    def test_batch_processing_mixed_file_types(self):
        """Test batch processing with mix of valid and invalid files."""
        files = [
            ("files", ("valid.jpg", self.test_image_bytes, "image/jpeg")),
            ("files", ("invalid.txt", b"not an image", "text/plain"))
        ]
        
        response = self.client.post("/process-batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should process valid files and skip invalid ones
        assert data["success_count"] == 1
        assert data["error_count"] == 1
        assert len(data["results"]) == 1
    
    @patch('src.core.document_processor.DocumentProcessor.process_document')
    def test_processing_performance_monitoring(self, mock_process):
        """Test that slow processing is logged."""
        # Mock slow processing (>100ms)
        from src.core.document_processor import DocumentResult
        mock_result = DocumentResult(
            angle=15.0,
            cropped_image=self.test_image,
            confidence=0.8,
            processing_time=0.15  # 150ms
        )
        mock_process.return_value = mock_result
        
        files = {"file": ("test.jpg", self.test_image_bytes, "image/jpeg")}
        
        with patch('src.api.main.logger') as mock_logger:
            response = self.client.post("/process-document", files=files)
            
            assert response.status_code == 200
            # Should log warning for slow processing
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "100ms threshold" in warning_call