import pytest
import numpy as np
import cv2
import tempfile
import os

@pytest.fixture
def sample_document_image():
    """Fixture providing a sample document image for testing."""
    # Create a simple document-like image
    image = np.ones((400, 300, 3), dtype=np.uint8) * 255
    
    # Add document border
    cv2.rectangle(image, (20, 20), (280, 380), (0, 0, 0), 2)
    
    # Add text-like lines
    for i in range(8):
        y = 50 + i * 40
        cv2.line(image, (40, y), (260, y), (0, 0, 0), 2)
        
        # Add some shorter lines
        if i % 2 == 0:
            cv2.line(image, (40, y + 15), (200, y + 15), (0, 0, 0), 1)
    
    return image

@pytest.fixture
def temp_image_file(sample_document_image):
    """Fixture providing a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, sample_document_image)
        yield tmp_file.name
    
    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)

@pytest.fixture
def rotated_document_image(sample_document_image):
    """Fixture providing a rotated document image."""
    height, width = sample_document_image.shape[:2]
    center = (width // 2, height // 2)
    
    # Rotate by 20 degrees
    rotation_matrix = cv2.getRotationMatrix2D(center, 20, 1.0)
    rotated = cv2.warpAffine(sample_document_image, rotation_matrix, (width, height))
    
    return rotated