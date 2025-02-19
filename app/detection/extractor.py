from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io
from app.models import ComparisonParams, DetectionParams
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class SignatureExtractor:
    """Handles signature extraction and feature calculation"""
    
    def __init__(self, 
                detection_params: Optional[DetectionParams] = None,
                comparison_params: Optional[ComparisonParams] = None):
        self.detection_params = detection_params or DetectionParams()
        self.comparison_params = comparison_params or ComparisonParams()

    async def extract_features(self, image_data: bytes) -> Dict[str, np.ndarray]:
        """
        Extract features from a signature image
        Returns dictionary of different feature types
        """
        try:
            # Convert bytes to image
            image = self._bytes_to_image(image_data)
            
            # Preprocess image
            processed = self._preprocess_image(image)
            
            # Extract all feature types
            features = {
                'hog': self._extract_hog_features(processed),
                'contour': self._extract_contour_features(processed),
                'density': self._extract_density_features(processed)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def _bytes_to_image(self, image_data: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format"""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for feature extraction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to standard size
        resized = cv2.resize(gray, settings.DEFAULT_IMAGE_SIZE)
        
        # Apply thresholding
        _, binary = cv2.threshold(
            resized,
            self.comparison_params.binary_threshold,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features"""
        win_size = settings.DEFAULT_IMAGE_SIZE
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride,
            cell_size, num_bins
        )
        
        features = hog.compute(image)
        return features.flatten()

    def _extract_contour_features(self, image: np.ndarray) -> np.ndarray:
        """Extract contour-based features"""
        # Find contours
        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        features = []
        
        for contour in contours:
            # Area
            area = cv2.contourArea(contour)
            
            # Perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            features.extend([
                area, perimeter, circularity,
                aspect_ratio, extent, solidity
            ])
        
        return np.array(features)

    def _extract_density_features(self, image: np.ndarray) -> np.ndarray:
        """Extract density grid features"""
        h, w = image.shape
        grid_size = self.comparison_params.grid_size
        
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Get cell coordinates
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                # Extract cell and calculate density
                cell = image[y_start:y_end, x_start:x_end]
                density = np.sum(cell == 0) / (cell_h * cell_w)
                
                features.append(density)
        
        return np.array(features)

    def extract_signature_image(self, 
                              image: np.ndarray,
                              region: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract signature region from image"""
        x, y, w, h = region
        padding = self.detection_params.padding
        
        # Add padding with bounds checking
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        return image[y_start:y_end, x_start:x_end]

    @staticmethod
    def save_signature(image: np.ndarray, 
                      path: str,
                      quality: int = 95) -> bool:
        """Save extracted signature to file"""
        try:
            success = cv2.imwrite(
                path,
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, quality]
            )
            return success
        except Exception as e:
            logger.error(f"Failed to save signature: {str(e)}")
            return False

async def extract_features(
    image_data: bytes,
    params: Optional[ComparisonParams] = None
) -> Dict[str, np.ndarray]:
    """
    Convenience function for one-off feature extraction
    """
    extractor = SignatureExtractor(comparison_params=params)
    return await extractor.extract_features(image_data)