from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import uuid
import pdf2image
from PIL import Image
import io
import logging
from pathlib import Path
import imutils

from app.models import DetectionParams, ProcessingResult, SignatureLocation
from app.config import settings
from app.utils.file_handler import save_temp_file

logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    """Handles document preprocessing and signature detection"""
    
    def __init__(self, params: Optional[DetectionParams] = None):
        self.params = params or DetectionParams()
        
    async def process_document(self,
                             content: bytes,
                             session_id: str,
                             is_pdf: bool) -> ProcessingResult:
        """Process document and detect signatures"""
        try:
            if is_pdf:
                images = self._pdf_to_images(content)
            else:
                images = [self._bytes_to_image(content)]
            
            results = []
            signature_counter = 1
            annotated_pages = []
            extracted_signatures = []
            
            for page_num, image in enumerate(images, 1):
                # Process each page
                preprocessed = self._preprocess_image(image)
                regions = self._detect_signature_regions(preprocessed, image)
                
                if regions:
                    # Create annotated image
                    annotated = image.copy()
                    
                    for region in regions:
                        # Extract and validate signature
                        if self._validate_signature_region(preprocessed, region):
                            x, y, w, h = region
                            
                            # Draw rectangle and ID
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                annotated,
                                f"#{signature_counter}",
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 255, 0),
                                2
                            )
                            
                            # Extract signature
                            signature = self._extract_signature(image, region)
                            
                            # Save signature image
                            signature_filename = f"{session_id}_signature_{signature_counter}.png"
                            signature_path = Path(settings.TEMP_DIR) / signature_filename
                            cv2.imwrite(str(signature_path), signature)
                            
                            extracted_signatures.append(signature_filename)
                            
                            # Store location
                            results.append(SignatureLocation(
                                page=page_num,
                                signature_id=signature_counter,
                                coordinates={
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h
                                }
                            ))
                            
                            signature_counter += 1
                    
                    # Save annotated page
                    page_filename = f"{session_id}_page_{page_num}.png"
                    page_path = Path(settings.TEMP_DIR) / page_filename
                    cv2.imwrite(str(page_path), annotated)
                    annotated_pages.append(page_filename)
            
            return ProcessingResult(
                total_signatures=signature_counter - 1,
                signatures=results,
                annotated_pages=annotated_pages,
                extracted_signatures=extracted_signatures
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

    def _pdf_to_images(self, content: bytes) -> List[np.ndarray]:
        """Convert PDF content to list of images"""
        # Create temporary file for PDF
        temp_dir = Path(settings.TEMP_DIR)
        temp_dir.mkdir(exist_ok=True)
        
        # Generate a unique filename
        pdf_path = temp_dir / f"{uuid.uuid4()}.pdf"
        
        try:
            # Write PDF content to file
            with open(pdf_path, "wb") as f:
                f.write(content)
            
            # Convert PDF pages to images
            pil_images = pdf2image.convert_from_path(
                str(pdf_path),
                dpi=200,
                size=settings.DEFAULT_IMAGE_SIZE
            )
            
            # Convert PIL images to OpenCV format
            return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
                for img in pil_images]
        finally:
            # Clean up temporary PDF
            pdf_path.unlink(missing_ok=True)

    def _bytes_to_image(self, content: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format"""
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for signature detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            gray,
            self.params.gaussian_blur_kernel,
            self.params.gaussian_blur_sigma
        )
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.params.adaptive_block_size,
            self.params.adaptive_c
        )
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def _detect_signature_regions(self,
                                preprocessed: np.ndarray,
                                original: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential signature regions"""
        # Find contours
        contours = cv2.findContours(
            preprocessed,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        signature_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.params.min_area:
                continue
            
            # Calculate perimeter and approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour,
                self.params.contour_approx_factor * peri,
                True
            )
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Calculate shape complexity
            complexity = peri * peri / (4 * np.pi * area)
            
            # Check aspect ratio
            aspect_ratio = w / h
            
            if (complexity > self.params.complexity_threshold and
                self.params.aspect_ratio_min < aspect_ratio < self.params.aspect_ratio_max):
                signature_regions.append((x, y, w, h))
        
        return signature_regions

    def _validate_signature_region(self,
                                 image: np.ndarray,
                                 region: Tuple[int, int, int, int]) -> bool:
        """Validate detected signature region"""
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        
        # Check ink density
        ink_density = np.sum(roi > 0) / (w * h)
        if ink_density < 0.01 or ink_density > 0.3:
            return False
        
        # Check for straight lines (usually not signatures)
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w/2)
        if lines is not None and len(lines) > 2:
            return False
        
        return True

    def _extract_signature(self,
                          image: np.ndarray,
                          region: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract signature region with padding"""
        x, y, w, h = region
        padding = self.params.padding
        
        # Add padding with bounds checking
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        return image[y_start:y_end, x_start:x_end]

async def process_document(content: bytes,
                         session_id: str,
                         is_pdf: bool,
                         params: Optional[DetectionParams] = None) -> ProcessingResult:
    """
    Convenience function for document processing
    """
    processor = DocumentPreprocessor(params)
    return await processor.process_document(content, session_id, is_pdf)