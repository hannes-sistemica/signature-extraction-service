import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional, Union, List
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to OpenCV format"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image data")
            
        return img
    except Exception as e:
        logger.error(f"Error converting bytes to CV2 image: {str(e)}")
        raise

def cv2_to_bytes(image: np.ndarray, format: str = '.png') -> bytes:
    """Convert OpenCV image to bytes"""
    try:
        success, buffer = cv2.imencode(format, image)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Error converting CV2 image to bytes: {str(e)}")
        raise

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        numpy_image = np.array(image)
        
        # Convert RGB to BGR
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error converting PIL to CV2 image: {str(e)}")
        raise

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    try:
        # Convert BGR to RGB
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        return Image.fromarray(color_converted)
    except Exception as e:
        logger.error(f"Error converting CV2 to PIL image: {str(e)}")
        raise

def resize_image(image: np.ndarray,
                size: Tuple[int, int],
                interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    try:
        h, w = image.shape[:2]
        target_w, target_h = size
        
        # Calculate aspect ratios
        aspect = w / h
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # Width is limiting factor
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            # Height is limiting factor
            new_h = target_h
            new_w = int(target_h * aspect)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Create canvas of target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate position to paste resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Paste resized image
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise

def normalize_image(image: np.ndarray,
                   target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Normalize image for processing"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if target size provided
        if target_size:
            gray = cv2.resize(gray, target_size)
        
        # Normalize pixel values
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        raise

def enhance_signature(image: np.ndarray) -> np.ndarray:
    """Enhance signature image for better visibility"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Invert back
        enhanced = cv2.bitwise_not(cleaned)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing signature: {str(e)}")
        raise

def crop_to_content(image: np.ndarray,
                    padding: int = 10) -> np.ndarray:
    """Crop image to content with padding"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to get content mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find content bounds
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop original image
        return image[y:y+h, x:x+w]
    except Exception as e:
        logger.error(f"Error cropping image to content: {str(e)}")
        raise

def rotate_image(image: np.ndarray,
                angle: float,
                center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Rotate image around center point"""
    try:
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image size
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        
        return rotated
    except Exception as e:
        logger.error(f"Error rotating image: {str(e)}")
        raise

def adjust_contrast(image: np.ndarray,
                   alpha: float = 1.5,
                   beta: int = 0) -> np.ndarray:
    """Adjust image contrast and brightness"""
    try:
        # Ensure alpha is positive
        alpha = max(0, alpha)
        
        # Apply contrast adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return adjusted
    except Exception as e:
        logger.error(f"Error adjusting contrast: {str(e)}")
        raise

def save_image(image: np.ndarray,
              path: Union[str, Path],
              quality: int = 95) -> bool:
    """Save image to file with quality settings"""
    try:
        path = Path(path)
        extension = path.suffix.lower()
        
        if extension not in ['.jpg', '.jpeg', '.png', '.tiff']:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Set compression parameters
        if extension in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif extension == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, quality // 10)]
        else:
            params = []
        
        # Save image
        success = cv2.imwrite(str(path), image, params)
        return success
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise