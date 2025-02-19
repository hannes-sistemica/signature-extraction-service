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
from app.config import settings  # Add explicit import

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
            logger.info(f"Processing document [session={session_id}]")
            if is_pdf:
                logger.info("Converting PDF document to images")
                images = self._pdf_to_images(content)
                logger.info(f"PDF conversion complete [pages={len(images)}]")
            else:
                logger.info("Processing single image document")
                images = [self._bytes_to_image(content)]
            
            results = []
            signature_counter = 1
            annotated_pages = []
            extracted_signatures = []
            
            logger.info(f"Starting page processing [total={len(images)}]")
            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page [number={page_num}]")
                # Process each page
                logger.info(f"Starting image preprocessing [page={page_num}]")
                preprocessed = self._preprocess_image(image)
                
                logger.info(f"Detecting signature regions [page={page_num}]")
                regions = self._detect_signature_regions(preprocessed, image)
                logger.info(f"Signature detection complete [regions={len(regions)}]")
                
                # Create annotated image only if signatures are found
                if regions:
                    annotated = image.copy()
                    valid_signatures = 0
                    
                    for region in regions:
                        # Extract and validate signature
                        if not self._validate_signature_region(preprocessed, region):
                            continue
                            
                        valid_signatures += 1
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
                
                    # Save annotated page only if valid signatures were found
                    if valid_signatures > 0:
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
            
            # Convert PDF pages to images at higher resolution
            pil_images = pdf2image.convert_from_path(
                str(pdf_path),
                dpi=300,  # Higher DPI for better quality
                first_page=1,
                last_page=settings.MAX_PDF_PAGES
            )
            logger.info(f"Converted PDF: found {len(pil_images)} pages")
            
            # Convert PIL images to OpenCV format
            cv_images = []
            for i, img in enumerate(pil_images, 1):
                logger.info(f"Converting page {i} to OpenCV format")
                cv_images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            return cv_images
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
        
        # Save debug image
        cv2.imwrite(str(Path(settings.TEMP_DIR) / "debug_01_gray.png"), gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        cv2.imwrite(str(Path(settings.TEMP_DIR) / "debug_02_thresh.png"), thresh)
        
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(str(Path(settings.TEMP_DIR) / "debug_03_cleaned.png"), cleaned)
        
        return cleaned

    def _detect_signature_regions(self,
                                preprocessed: np.ndarray,
                                original: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential signature regions"""
        # Find horizontal and vertical lines
        horizontal = preprocessed.copy()
        vertical = preprocessed.copy()
        
        # Adjust these values based on your image size
        h_size = horizontal.shape[1] // 30
        v_size = vertical.shape[0] // 30
        
        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        
        # Remove horizontal and vertical lines
        horizontal = cv2.erode(horizontal, h_structure)
        horizontal = cv2.dilate(horizontal, h_structure)
        vertical = cv2.erode(vertical, v_structure)
        vertical = cv2.dilate(vertical, v_structure)
        
        # Remove lines from original
        mask = cv2.bitwise_or(horizontal, vertical)
        cv2.imwrite(str(Path(settings.TEMP_DIR) / "debug_04_lines_mask.png"), mask)
        
        cleaned = cv2.bitwise_xor(preprocessed, mask)
        cv2.imwrite(str(Path(settings.TEMP_DIR) / "debug_05_no_lines.png"), cleaned)
        
        # Find contours in cleaned image
        contours, _ = cv2.findContours(
            cleaned, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        signature_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 500:
                continue
                
            complexity = perimeter * perimeter / (4 * np.pi * area)
            
            if complexity > 20:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if region is isolated (not part of text block)
                roi = cleaned[max(0, y-20):min(cleaned.shape[0], y+h+20),
                            max(0, x-20):min(cleaned.shape[1], x+w+20)]
                
                if roi.size > 0:
                    text_density = np.sum(roi > 0) / roi.size
                    
                    # Save debug ROI
                    debug_roi = cleaned.copy()
                    cv2.rectangle(debug_roi, (x, y), (x + w, y + h), 255, 2)
                    cv2.putText(debug_roi, f"d={text_density:.2f} r={aspect_ratio:.2f}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                    cv2.imwrite(str(Path(settings.TEMP_DIR) / f"debug_06_region_{x}_{y}.png"), debug_roi)
                    
                    # Signatures typically have lower density than text blocks
                    if text_density < 0.2 and 0.5 < aspect_ratio < 5:
                        signature_regions.append((x, y, w, h))
        
        return signature_regions

    def _validate_signature_region(self,
                                 image: np.ndarray,
                                 region: Tuple[int, int, int, int]) -> bool:
        """Validate detected signature region"""
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        
        # Log validation steps
        logger.info(f"Validating region: x={x}, y={y}, w={w}, h={h}")
        
        # Calculate basic metrics
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.info("Region rejected: No contours found")
            return False
            
        # Get the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate shape complexity metrics
        complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
        solidity = area / hull_area if hull_area > 0 else 0
        
        logger.info(f"Shape complexity: {complexity:.2f}, Solidity: {solidity:.2f}")
        
        # Check minimum size
        if w < 30 or h < 15:
            logger.info("Region rejected: Too small")
            return False
            
        # Check aspect ratio
        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 7:
            logger.info(f"Region rejected: Bad aspect ratio ({aspect_ratio})")
            return False
        
        # Check ink density with more tolerance for signatures
        ink_density = np.sum(roi > 0) / (w * h)
        logger.info(f"Ink density: {ink_density:.2f}")
        if ink_density < 0.02 or ink_density > 0.7:  # More tolerant upper bound
            logger.info("Region rejected: Invalid ink density")
            return False
            
        # Check shape complexity (signatures are typically complex)
        if complexity < 15 or complexity > 100:  # Adjusted thresholds
            logger.info("Region rejected: Invalid shape complexity")
            return False
            
        # Check solidity (signatures typically have moderate solidity)
        if solidity < 0.1 or solidity > 0.95:
            logger.info("Region rejected: Invalid solidity")
            return False
        
        # Check for too many straight lines
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w/2)
        if lines is not None and len(lines) > 5:
            logger.info("Region rejected: Too many straight lines")
            return False
        
        logger.info("Region validated as signature")
        return True

    def _extract_signature(self,
                          image: np.ndarray,
                          region: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract signature region with padding"""
        x, y, w, h = region
        padding = settings.PADDING
        
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
