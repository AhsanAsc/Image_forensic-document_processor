# src/core/document_processor.py
import cv2
import numpy as np
from typing import Tuple, Optional
import time
from dataclasses import dataclass

@dataclass
class DocumentResult:
    angle: float
    cropped_image: np.ndarray
    confidence: float
    processing_time: float

class DocumentProcessor:
    """Main document processing class that combines deskewing and boundary detection."""
    
    def __init__(self, max_dimension: int = 1024, padding: int = 10):
        self.max_dimension = max_dimension
        self.padding = padding
        
    def process_document(self, image: np.ndarray) -> DocumentResult:
        """Process document: deskew and crop boundaries."""
        start_time = time.time()
        
        # Store original dimensions for scaling back
        original_height, original_width = image.shape[:2]
        
        # Resize for processing if needed
        processed_image, scale_factor = self._resize_for_processing(image)
        
        # 1. Deskewing
        angle, deskewed_image = self._deskew_document(processed_image)
        
        # 2. Boundary detection and cropping
        cropped_image = self._detect_and_crop_boundaries(deskewed_image)
        
        # Scale back to appropriate size
        if scale_factor != 1.0:
            target_height = int(cropped_image.shape[0] / scale_factor)
            target_width = int(cropped_image.shape[1] / scale_factor)
            cropped_image = cv2.resize(cropped_image, (target_width, target_height), 
                                     interpolation=cv2.INTER_CUBIC)
        
        # Add consistent padding
        cropped_image = self._add_padding(cropped_image, self.padding)
        
        processing_time = time.time() - start_time
        confidence = self._calculate_confidence(angle, cropped_image)
        
        return DocumentResult(
            angle=angle,
            cropped_image=cropped_image,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _resize_for_processing(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize image for processing while maintaining aspect ratio."""
        height, width = image.shape[:2]
        
        if max(height, width) <= self.max_dimension:
            return image, 1.0
        
        scale_factor = self.max_dimension / max(height, width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale_factor
    
    def _deskew_document(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect and correct document rotation using multiple methods."""
        
        # Method 1: Hough Line Transform (primary)
        angle_hough = self._detect_angle_hough_lines(image)
        
        # Method 2: Projection Profile (backup)
        angle_projection = self._detect_angle_projection_profile(image)
        
        # Choose the best angle based on confidence
        angle = self._select_best_angle(angle_hough, angle_projection, image)
        
        # Apply rotation
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            deskewed = self._rotate_image(image, angle)
        else:
            deskewed = image.copy()
            angle = 0.0
            
        return angle, deskewed
    
    def _detect_angle_hough_lines(self, image: np.ndarray) -> float:
        """Detect rotation angle using Hough line transform."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
        
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            # Normalize angle to [-45, 45] range
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Use median for robustness
        return np.median(angles)
    
    def _detect_angle_projection_profile(self, image: np.ndarray) -> float:
        """Detect rotation angle using projection profile analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        max_variance = 0
        best_angle = 0
        
        # Test angles from -45 to 45 degrees
        for angle in range(-45, 46, 1):
            rotated = self._rotate_image(binary, angle)
            
            # Calculate horizontal projection
            h_projection = np.sum(rotated, axis=1)
            variance = np.var(h_projection)
            
            if variance > max_variance:
                max_variance = variance
                best_angle = angle
        
        return best_angle
    
    def _select_best_angle(self, angle1: float, angle2: float, image: np.ndarray) -> float:
        """Select the best angle from multiple detection methods."""
        # If angles are close, prefer Hough lines (usually more accurate)
        if abs(angle1 - angle2) < 5:
            return angle1
        
        # Otherwise, use additional validation
        # For now, prefer Hough lines if available
        if abs(angle1) < 45:
            return angle1
        return angle2
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle with proper handling of boundaries."""
        if abs(angle) < 0.1:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        return rotated

    def _find_best_document_quad_lsd(self, image: np.ndarray) -> Optional[np.ndarray]:
        # --- prep ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # --- detect line segments ---
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(gray)[0]
        if lines is None or len(lines) < 4:
            return None

        # Flatten & score lines by length
        L = []
        for ln in lines.reshape(-1,4):
            x1,y1,x2,y2 = ln
            length = np.hypot(x2-x1, y2-y1)
            if length > 0:
                ang = np.degrees(np.arctan2((y2-y1), (x2-x1))) % 180.0
                L.append((x1,y1,x2,y2,length,ang))
        if not L:
            return None

        # Keep top-N longest lines to reduce combinatorics
        L.sort(key=lambda t: t[4], reverse=True)
        L = L[:80]

        # Cluster angles modulo 90 to get two main directions
        angs = np.array([((a+90)%90) for *_, a in L], dtype=np.float32).reshape(-1,1)
        if len(angs) < 2:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        K = 2
        compactness, labels, centers = cv2.kmeans(angs, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        # split lines into two orientation groups
        g0 = [L[i] for i in range(len(L)) if labels[i]==0]
        g1 = [L[i] for i in range(len(L)) if labels[i]==1]
        if len(g0) < 2 or len(g1) < 2:
            return None

        # helper: line (p1->p2) in ax+by+c=0 form
        def line_abc(x1,y1,x2,y2):
            a = y1 - y2
            b = x2 - x1
            c = x1*y2 - x2*y1
            nrm = np.hypot(a,b)
            if nrm > 0: a,b,c = a/nrm, b/nrm, c/nrm
            return a,b,c

        # helper: intersection of two lines (ax+by+c=0)
        def intersect(l1, l2):
            a1,b1,c1 = l1
            a2,b2,c2 = l2
            d = a1*b2 - a2*b1
            if abs(d) < 1e-6: return None
            x = (b1*c2 - b2*c1)/d
            y = (c1*a2 - c2*a1)/d
            return np.array([x,y], dtype=np.float32)

        # Convert each segment to its infinite line form
        G0 = [(*t, line_abc(t[0],t[1],t[2],t[3])) for t in g0]
        G1 = [(*t, line_abc(t[0],t[1],t[2],t[3])) for t in g1]

        H, W = image.shape[:2]
        img_area = float(H*W)

        best = None  # (score, quad4x2)
        # Use top M from each group to limit pairs
        M0, M1 = min(15,len(G0)), min(15,len(G1))

        for i in range(M0):
            for j in range(i+1, M0):
                l0a = G0[i][-1]; l0b = G0[j][-1]
                # roughly parallel? enforce small angle between them
                # since in same cluster, okay.

                for p in range(M1):
                    for q in range(p+1, M1):
                        l1a = G1[p][-1]; l1b = G1[q][-1]

                        # Intersections produce 4 corners:
                        P00 = intersect(l0a, l1a)
                        P01 = intersect(l0a, l1b)
                        P10 = intersect(l0b, l1a)
                        P11 = intersect(l0b, l1b)
                        if P00 is None or P01 is None or P10 is None or P11 is None:
                            continue
                        quad = np.stack([P00,P01,P11,P10], axis=0)  # order rough TL,TR,BR,BL
                        # sanity: all points inside a padded image bounds
                        if np.any(np.isnan(quad)) or np.any(np.isinf(quad)):
                            continue
                        # area & shape checks
                        ordered = self._order_corners(quad)
                        w_top = np.linalg.norm(ordered[1]-ordered[0])
                        w_bot = np.linalg.norm(ordered[2]-ordered[3])
                        h_lft = np.linalg.norm(ordered[3]-ordered[0])
                        h_rgt = np.linalg.norm(ordered[2]-ordered[1])
                        w = max(w_top, w_bot); h = max(h_lft, h_rgt)
                        if w < 40 or h < 40:
                            continue
                        ar = w/h if w>=h else h/max(w,1e-6)
                        if not (1.2 <= ar <= 1.9):
                            continue

                        # polygon area
                        area = cv2.contourArea(ordered.astype(np.float32))
                        if area < 0.15*img_area or area > 0.95*img_area:
                            continue

                        # score: area + side support by nearby segments
                        # measure how many original segments lie close to each side
                        def side_support(P,Q, tol=5.0):
                            a,b,c = line_abc(P[0],P[1],Q[0],Q[1])
                            support = 0.0
                            for (x1,y1,x2,y2,ln,_ang) in L:
                                # distance of endpoints to line
                                d1 = abs(a*x1 + b*y1 + c)
                                d2 = abs(a*x2 + b*y2 + c)
                                if min(d1,d2) < tol:
                                    support += max(1.0, np.hypot(x2-x1,y2-y1)/100.0)
                            return support

                        s1 = side_support(ordered[0], ordered[1])
                        s2 = side_support(ordered[1], ordered[2])
                        s3 = side_support(ordered[2], ordered[3])
                        s4 = side_support(ordered[3], ordered[0])
                        support = (s1+s2+s3+s4)/4.0

                        # center bias: prefer quads near image center
                        cx, cy = ordered.mean(axis=0)
                        cxn, cyn = cx/W, cy/H
                        center_bonus = 1.0 - np.linalg.norm([cxn-0.5, cyn-0.5])

                        score = 0.6*(area/img_area) + 0.3*(support/10.0) + 0.1*center_bonus
                        if best is None or score > best[0]:
                            best = (score, ordered)

        if best is None or best[0] < 0.12:
            return None
        return best[1]

    def _detect_and_crop_boundaries_contour(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
        if np.mean(th) < 127: th = cv2.bitwise_not(th)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=2)
        edges = cv2.Canny(th, 50, 150)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return image

        H,W = image.shape[:2]; img_area = float(H*W)
        best = None

        for idx,cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 0.12*img_area or area > 0.98*img_area:
                continue
            # bounding box + rectangularity
            x,y,w,h = cv2.boundingRect(cnt)
            rect_area = float(w*h)
            if rect_area <= 0: continue
            rectangularity = area/rect_area
            if rectangularity < 0.7:  # suppress wavy blobs
                continue
            ar = w/h if w>=h else h/max(w,1e-6)
            if not (1.2 <= ar <= 1.9):
                continue

            # polygon approx
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            is_quad = (len(approx)==4 and cv2.isContourConvex(approx))
            score = 0.5*(area/img_area) + 0.3*rectangularity + 0.2*(1.0 if is_quad else 0.0)

            if best is None or score > best[0]:
                best = (score, approx if is_quad else cv2.minAreaRect(cnt), is_quad)

        if best is None: 
            return image
        score, shape, is_quad = best
        if is_quad: 
            return self._crop_using_corners(image, shape)
        else:
            return self._warp_from_rotated_rect(image, shape)


    def _detect_and_crop_boundaries(self, image: np.ndarray) -> np.ndarray:
        # 1) Try line-based quad detection first
        quad = self._find_best_document_quad_lsd(image)
        if quad is not None:
            return self._crop_using_corners(image, quad.astype(np.float32))

        # 2) Fallback to improved contour method (your existing, but stricter filters)
        return self._detect_and_crop_boundaries_contour(image)

    
    def _crop_using_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Crop image using detected corners with perspective correction."""
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners.reshape(4, 2))
        
        # Calculate target dimensions
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        max_width = max(int(width_top), int(width_bottom))
        
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        max_height = max(int(height_left), int(height_right))
        
        # Define destination points
        dst_corners = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Apply perspective transform
        transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
        cropped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
        
        return cropped
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as top-left, top-right, bottom-right, bottom-left."""
        # Sum coordinates
        s = corners.sum(axis=1)
        # Top-left has smallest sum, bottom-right has largest
        top_left = corners[np.argmin(s)]
        bottom_right = corners[np.argmax(s)]
        
        # Difference coordinates
        diff = np.diff(corners, axis=1)
        # Top-right has smallest difference, bottom-left has largest
        top_right = corners[np.argmin(diff)]
        bottom_left = corners[np.argmax(diff)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left])
    
    def _crop_using_bounding_rect(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Crop image using bounding rectangle as fallback."""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add some margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        return image[y:y+h, x:x+w]
    
    def _add_padding(self, image: np.ndarray, padding: int) -> np.ndarray:
        """Add consistent padding around the image."""
        if padding <= 0:
            return image
        
        return cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    def _calculate_confidence(self, angle: float, cropped_image: np.ndarray) -> float:
        """Calculate confidence score for the processing result."""
        # Simple confidence based on angle certainty and image quality
        angle_confidence = max(0, 1 - abs(angle) / 45)  # Higher confidence for smaller angles
        
        # Image quality metrics (contrast, sharpness, etc.)
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image
        contrast = np.std(gray) / 255.0
        
        # Combine metrics
        confidence = (angle_confidence * 0.6) + (contrast * 0.4)
        return min(1.0, max(0.0, confidence))


# Example usage and testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Load sample image
    image = cv2.imread("sample_document.jpg")
    
    if image is not None:
        result = processor.process_document(image)
        
        print(f"Detected angle: {result.angle:.2f}°")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing time: {result.processing_time*1000:.1f}ms")
        
        # Save result
        cv2.imwrite("processed_document.jpg", result.cropped_image)
    else:
        print("Sample image not found!")