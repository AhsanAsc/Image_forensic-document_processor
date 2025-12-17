import pytest
import numpy as np
import cv2
import time
from unittest.mock import Mock, patch
from src.core.document_processor import DocumentProcessor, DocumentResult

class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = DocumentProcessor()
        
        # Create a simple test image (white rectangle on black background)
        self.test_image = self.create_test_document_image()
        
        # Create a rotated test image
        self.rotated_image = self.create_rotated_test_image(15.0)
    
    def create_test_document_image(self, width=400, height=300):
        """Create a simple test document image."""
        # Black background
        image = np.zeros((height + 100, width + 100, 3), dtype=np.uint8)
        
        # White document rectangle
        cv2.rectangle(image, (50, 50), (50 + width, 50 + height), (255, 255, 255), -1)
        
        # Add some text-like lines
        for i in range(5):
            y = 80 + i * 40
            cv2.line(image, (70, y), (width + 30, y), (0, 0, 0), 2)
        
        return image
    
    def create_rotated_test_image(self, angle):
        """Create a rotated test document image."""
        image = self.create_test_document_image()
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(max_dimension=512, padding=20)
        assert processor.max_dimension == 512
        assert processor.padding == 20
    
    def test_process_document_returns_correct_structure(self):
        """Test that process_document returns DocumentResult with correct structure."""
        result = self.processor.process_document(self.test_image)
        
        assert isinstance(result, DocumentResult)
        assert hasattr(result, 'angle')
        assert hasattr(result, 'cropped_image')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_time')
        
        assert isinstance(result.angle, float)
        assert isinstance(result.cropped_image, np.ndarray)
        assert isinstance(result.confidence, float)
        assert isinstance(result.processing_time, float)
    
    def test_process_document_performance(self):
        """Test that processing time is within acceptable limits."""
        start_time = time.time()
        result = self.processor.process_document(self.test_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should be under 100ms for most cases
        assert processing_time < 0.2  # 200ms allowance for testing environment
        assert result.processing_time > 0
    
    def test_deskew_detection_accuracy(self):
        """Test that deskewing detects angles accurately."""
        # Test with known rotation
        result = self.processor.process_document(self.rotated_image)
        
        # Should detect rotation within reasonable tolerance
        detected_angle = abs(result.angle)
        assert 10.0 <= detected_angle <= 20.0, f"Expected angle ~15°, got {detected_angle}°"
    
    def test_resize_for_processing(self):
        """Test image resizing functionality."""
        # Large image
        large_image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255
        
        resized, scale_factor = self.processor._resize_for_processing(large_image)
        
        assert max(resized.shape[:2]) <= self.processor.max_dimension
        assert scale_factor < 1.0
        
        # Small image (should not be resized)
        small_image = np.ones((300, 200, 3), dtype=np.uint8) * 255
        resized_small, scale_small = self.processor._resize_for_processing(small_image)
        
        assert np.array_equal(resized_small, small_image)
        assert scale_small == 1.0
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        result = self.processor.process_document(self.test_image)
        
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence > 0.1  # Should have reasonable confidence
    
    def test_padding_addition(self):
        """Test consistent padding addition."""
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        padding = 15
        
        padded = self.processor._add_padding(test_img, padding)
        
        expected_height = test_img.shape[0] + 2 * padding
        expected_width = test_img.shape[1] + 2 * padding
        
        assert padded.shape[0] == expected_height
        assert padded.shape[1] == expected_width
        
        # Check padding color (should be white)
        assert np.all(padded[0, :] == [255, 255, 255])  # Top edge
        assert np.all(padded[-1, :] == [255, 255, 255])  # Bottom edge
    
    def test_error_handling_empty_image(self):
        """Test handling of invalid/empty images."""
        # Empty image
        empty_image = np.array([])
        
        with pytest.raises(Exception):
            self.processor.process_document(empty_image)
    
    def test_grayscale_image_handling(self):
        """Test processing of grayscale images."""
        # Convert test image to grayscale
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        
        result = self.processor.process_document(gray_image)
        
        assert isinstance(result, DocumentResult)
        assert result.cropped_image.shape[0] > 0
        assert result.cropped_image.shape[1] > 0
    
    def test_angle_normalization(self):
        """Test that detected angles are properly normalized."""
        # Test multiple rotated images
        angles_to_test = [-30, -15, 0, 15, 30, 45]
        
        for angle in angles_to_test:
            rotated = self.create_rotated_test_image(angle)
            result = self.processor.process_document(rotated)
            
            # Detected angle should be within reasonable range
            assert -45 <= result.angle <= 45
    
    @patch('cv2.HoughLines')
    def test_fallback_to_projection_profile(self, mock_hough):
        """Test fallback to projection profile when Hough lines fail."""
        # Mock HoughLines to return None (failure)
        mock_hough.return_value = None
        
        result = self.processor.process_document(self.rotated_image)
        
        # Should still process successfully using backup method
        assert isinstance(result, DocumentResult)
        mock_hough.assert_called()