# examples/sample_usage.py
"""
Sample usage examples for the Document Processing System.
Demonstrates various ways to use the processor with performance monitoring.
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
import glob
from typing import List, Dict, Any

from src.core.document_processor import DocumentProcessor, DocumentResult


class ProcessingBenchmark:
    """Utility class for benchmarking document processing performance."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.results: List[Dict[str, Any]] = []
    
    def process_single_image(self, image_path: str, verbose: bool = True) -> DocumentResult:
        """Process a single image and log performance metrics."""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        if verbose:
            print(f"\nProcessing: {Path(image_path).name}")
            print(f"Original size: {orig_width}x{orig_height} ({file_size_mb:.1f}MB)")
        
        # Process document
        start_time = time.time()
        result = self.processor.process_document(image)
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        processing_speed = (orig_width * orig_height) / (total_time * 1000000)  # Megapixels per second
        
        # Store results
        benchmark_data = {
            'file_path': image_path,
            'original_size': (orig_width, orig_height),
            'file_size_mb': file_size_mb,
            'detected_angle': result.angle,
            'confidence': result.confidence,
            'processing_time_ms': total_time * 1000,
            'processing_speed_mpx_s': processing_speed,
            'meets_target': total_time < 0.1  # 100ms target
        }
        self.results.append(benchmark_data)
        
        if verbose:
            print(f"Detected angle: {result.angle:.2f}°")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Processing time: {total_time*1000:.1f}ms {'✓' if total_time < 0.1 else '⚠️'}")
            print(f"Speed: {processing_speed:.1f} MP/s")
        
        return result
    
    def process_batch(self, image_paths: List[str], save_results: bool = False) -> Dict[str, Any]:
        """Process multiple images and generate performance report."""
        
        print(f"Processing batch of {len(image_paths)} images...")
        
        successful = 0
        failed = 0
        total_time = 0
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"\n[{i}/{len(image_paths)}]", end=" ")
                result = self.process_single_image(image_path, verbose=True)
                
                if save_results:
                    output_path = f"processed_{Path(image_path).stem}.jpg"
                    cv2.imwrite(output_path, result.cropped_image)
                    print(f"Saved: {output_path}")
                
                successful += 1
                total_time += self.results[-1]['processing_time_ms']
                
            except Exception as e:
                print(f"Failed to process {image_path}: {str(e)}")
                failed += 1
        
        # Generate summary report
        if self.results:
            avg_time = np.mean([r['processing_time_ms'] for r in self.results])
            avg_confidence = np.mean([r['confidence'] for r in self.results])
            target_met_count = sum([r['meets_target'] for r in self.results])
            
            report = {
                'total_images': len(image_paths),
                'successful': successful,
                'failed': failed,
                'average_time_ms': avg_time,
                'average_confidence': avg_confidence,
                'target_met_percentage': (target_met_count / successful * 100) if successful > 0 else 0,
                'total_batch_time_ms': total_time
            }
            
            print(f"\n{'='*50}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Total images: {report['total_images']}")
            print(f"Successful: {report['successful']}")
            print(f"Failed: {report['failed']}")
            print(f"Average processing time: {report['average_time_ms']:.1f}ms")
            print(f"Average confidence: {report['average_confidence']:.3f}")
            print(f"Target met (<100ms): {report['target_met_percentage']:.1f}%")
            print(f"Total batch time: {report['total_batch_time_ms']:.1f}ms")
            
            return report
        
        return {}
    
    def generate_performance_report(self, output_file: str = "performance_report.txt"):
        """Generate detailed performance report."""
        
        if not self.results:
            print("No results to report.")
            return
        
        with open(output_file, 'w') as f:
            f.write("DOCUMENT PROCESSOR PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            times = [r['processing_time_ms'] for r in self.results]
            confidences = [r['confidence'] for r in self.results]
            angles = [r['detected_angle'] for r in self.results]
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total images processed: {len(self.results)}\n")
            f.write(f"Average processing time: {np.mean(times):.1f}ms\n")
            f.write(f"Median processing time: {np.median(times):.1f}ms\n")
            f.write(f"Min/Max processing time: {np.min(times):.1f}ms / {np.max(times):.1f}ms\n")
            f.write(f"Target met (<100ms): {sum([t < 100 for t in times])}/{len(times)} ({sum([t < 100 for t in times])/len(times)*100:.1f}%)\n")
            f.write(f"Average confidence: {np.mean(confidences):.3f}\n")
            f.write(f"Average detected angle: {np.mean(np.abs(angles)):.2f}°\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'File':<30} {'Time(ms)':<10} {'Angle(°)':<10} {'Confidence':<12} {'Target Met':<12}\n")
            f.write("-" * 74 + "\n")
            
            for result in self.results:
                filename = Path(result['file_path']).name
                f.write(f"{filename:<30} {result['processing_time_ms']:<10.1f} {result['detected_angle']:<10.2f} "
                       f"{result['confidence']:<12.3f} {'Yes' if result['meets_target'] else 'No':<12}\n")
        
        print(f"Performance report saved to: {output_file}")


def create_test_dataset(output_dir: str = "test_images", count: int = 5):
    """Create a test dataset with various document scenarios."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating test dataset in: {output_dir}")
    
    for i in range(count):
        # Create base document
        height, width = 400, 300
        image = np.ones((height + 200, width + 200, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add document rectangle
        doc_color = (255, 255, 255)  # White document
        cv2.rectangle(image, (100, 100), (100 + width, 100 + height), doc_color, -1)
        cv2.rectangle(image, (100, 100), (100 + width, 100 + height), (0, 0, 0), 2)  # Border
        
        # Add text lines
        for j in range(12):
            y = 130 + j * 25
            line_width = np.random.randint(150, 250)  # Vary line lengths
            cv2.line(image, (120, y), (120 + line_width, y), (0, 0, 0), 2)
            
            # Add some shorter lines occasionally
            if j % 3 == 0:
                cv2.line(image, (120, y + 10), (120 + line_width - 50, y + 10), (0, 0, 0), 1)
        
        # Apply random rotation
        angle = np.random.uniform(-30, 30)  # Random angle between -30 and 30 degrees
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        # Add some noise/artifacts occasionally
        if i % 3 == 0:
            # Add shadow
            shadow_overlay = rotated.copy()
            cv2.ellipse(shadow_overlay, (center[0] + 50, center[1] + 50), (200, 100), 0, 0, 360, (200, 200, 200), -1)
            rotated = cv2.addWeighted(rotated, 0.8, shadow_overlay, 0.2, 0)
        
        # Save image
        filename = f"{output_dir}/test_document_{i+1:02d}_angle_{angle:.1f}.jpg"
        cv2.imwrite(filename, rotated)
        print(f"Created: {filename}")
    
    print(f"Test dataset created with {count} images.")
    return glob.glob(f"{output_dir}/*.jpg")


def performance_optimization_demo():
    """Demonstrate performance optimization techniques."""
    
    print("PERFORMANCE OPTIMIZATION DEMO")
    print("=" * 40)
    
    # Create processor with different configurations
    configs = [
        ("Standard", DocumentProcessor()),
        ("High Speed", DocumentProcessor(max_dimension=512, padding=5)),
        ("High Quality", DocumentProcessor(max_dimension=1536, padding=20))
    ]
    
    # Create test image
    test_image = np.ones((1200, 800, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (100, 100), (700, 1100), (0, 0, 0), 2)
    for i in range(20):
        y = 150 + i * 40
        cv2.line(test_image, (150, y), (650, y), (0, 0, 0), 2)
    
    # Test each configuration
    results = {}
    for name, processor in configs:
        print(f"\nTesting {name} configuration...")
        times = []
        
        # Run multiple times for average
        for _ in range(5):
            start_time = time.time()
            result = processor.process_document(test_image)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        results[name] = {
            'avg_time_ms': avg_time,
            'meets_target': avg_time < 100
        }
        
        print(f"Average time: {avg_time:.1f}ms {'✓' if avg_time < 100 else '⚠️'}")
    
    # Show comparison
    print(f"\n{'Configuration':<15} {'Avg Time (ms)':<15} {'Meets Target':<15}")
    print("-" * 45)
    for name, data in results.items():
        status = "Yes" if data['meets_target'] else "No"
        print(f"{name:<15} {data['avg_time_ms']:<15.1f} {status:<15}")


def main():
    """Main demonstration function."""
    
    print("DOCUMENT PROCESSING SYSTEM DEMO")
    print("=" * 50)
    
    # 1. Create test dataset
    print("\n1. Creating test dataset...")
    test_images = create_test_dataset("demo_images", 3)
    
    # 2. Performance optimization demo
    print("\n2. Performance optimization comparison...")
    performance_optimization_demo()
    
    # 3. Batch processing demo
    print("\n3. Batch processing demo...")
    benchmark = ProcessingBenchmark()
    report = benchmark.process_batch(test_images, save_results=True)
    
    # 4. Generate performance report
    print("\n4. Generating performance report...")
    benchmark.generate_performance_report("demo_performance_report.txt")
    
    # 5. API usage example
    print("\n5. API usage example...")
    print("To start the API server, run:")
    print("  poetry run uvicorn src.api.main:app --reload")
    print("\nThen test with:")
    print("  curl -X POST 'http://localhost:8000/process-document' \\")
    print("    -H 'Content-Type: multipart/form-data' \\")
    print("    -F 'file=@demo_images/test_document_01_angle_*.jpg'")
    
    print(f"\nDemo completed! Check the generated files:")
    print(f"- Processed images: processed_*.jpg")
    print(f"- Performance report: demo_performance_report.txt")


if __name__ == "__main__":
    main()


# Performance optimization tips and advanced usage patterns

"""
PERFORMANCE OPTIMIZATION GUIDE
==============================

1. IMAGE PREPROCESSING OPTIMIZATIONS:
   - Resize large images before processing (max_dimension parameter)
   - Use appropriate color spaces (grayscale when possible)
   - Apply noise reduction only when needed

2. ALGORITHM SELECTION:
   - Hough Line Transform: Fast, good for documents with clear text lines
   - Projection Profile: Better for dense text, but slower
   - Use confidence scores to select best method

3. MEMORY MANAGEMENT:
   - Avoid unnecessary image copies
   - Process images in batches for large datasets
   - Use appropriate data types (uint8 vs float64)

4. PARALLELIZATION STRATEGIES:
   - Process multiple images concurrently
   - Use multiprocessing for CPU-intensive tasks
   - Consider GPU acceleration for large-scale processing

5. CACHING OPTIMIZATIONS:
   - Cache preprocessing results
   - Store computed parameters for similar images
   - Use lookup tables for common transformations

ADVANCED USAGE PATTERNS:
========================

1. Custom Configuration:
   processor = DocumentProcessor(
       max_dimension=1024,  # Balance speed vs quality
       padding=10          # Adjust based on requirements
   )

2. Confidence-based Processing:
   result = processor.process_document(image)
   if result.confidence < 0.7:
       # Try alternative processing or manual review
       pass

3. Multi-scale Processing:
   # Process at different resolutions for robustness
   scales = [512, 1024, 1536]
   best_result = None
   best_confidence = 0
   
   for scale in scales:
       proc = DocumentProcessor(max_dimension=scale)
       result = proc.process_document(image)
       if result.confidence > best_confidence:
           best_result = result
           best_confidence = result.confidence

4. Batch Processing with Error Recovery:
   def robust_batch_process(image_paths):
       results = []
       for path in image_paths:
           try:
               result = processor.process_document(cv2.imread(path))
               results.append((path, result, None))
           except Exception as e:
               results.append((path, None, str(e)))
       return results

5. Integration with ML Pipelines:
   # Use as preprocessing step in ML pipeline
   def preprocess_for_ocr(image_path):
       image = cv2.imread(image_path)
       result = processor.process_document(image)
       
       # Further preprocessing for OCR
       processed = cv2.cvtColor(result.cropped_image, cv2.COLOR_BGR2GRAY)
       processed = cv2.adaptiveThreshold(processed, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
       return processed

TROUBLESHOOTING COMMON ISSUES:
==============================

1. Poor Angle Detection:
   - Check image contrast and quality
   - Ensure document has clear text lines
   - Try different preprocessing approaches

2. Boundary Detection Failures:
   - Verify sufficient contrast between document and background
   - Check for complete document visibility
   - Consider manual ROI selection as fallback

3. Performance Issues:
   - Reduce max_dimension for faster processing
   - Profile code to identify bottlenecks
   - Consider hardware acceleration options

4. Memory Usage:
   - Monitor peak memory usage during batch processing
   - Implement proper cleanup for large images
   - Use generators for large datasets
"""