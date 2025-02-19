from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from typing import List, Dict
import cv2
import numpy as np
import pdf2image
import os
from pydantic import BaseModel
import shutil
from pathlib import Path
import uuid
import tempfile
import time

app = FastAPI(
    title="Signature Detection API",
    description="""
    This API provides signature detection capabilities for PDF documents.
    It can detect signatures, extract them, and provide annotated versions of the pages.
    
    ## Features
    * Detect signatures in PDF documents
    * Extract individual signatures as separate images
    * Provide annotated PDFs showing signature locations
    * Number signatures sequentially across all pages
    
    ## Usage
    1. Upload a PDF using the `/detect-signatures/` endpoint
    2. Retrieve detected signatures and annotated pages using the `/files/{filename}` endpoint
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

class SignatureLocation(BaseModel):
    """
    Represents the location of a detected signature
    """
    page: int
    signature_id: int
    coordinates: Dict[str, int]

    class Config:
        schema_extra = {
            "example": {
                "page": 1,
                "signature_id": 1,
                "coordinates": {
                    "x": 100,
                    "y": 200,
                    "width": 300,
                    "height": 100
                }
            }
        }

class ProcessingResult(BaseModel):
    """
    Contains the results of processing a PDF for signatures
    """
    total_signatures: int
    signatures: List[SignatureLocation]
    annotated_pages: List[str]
    extracted_signatures: List[str]

    class Config:
        schema_extra = {
            "example": {
                "total_signatures": 2,
                "signatures": [
                    {
                        "page": 1,
                        "signature_id": 1,
                        "coordinates": {
                            "x": 100,
                            "y": 200,
                            "width": 300,
                            "height": 100
                        }
                    },
                    {
                        "page": 2,
                        "signature_id": 2,
                        "coordinates": {
                            "x": 150,
                            "y": 250,
                            "width": 280,
                            "height": 90
                        }
                    }
                ],
                "annotated_pages": ["session_123_page_1.png", "session_123_page_2.png"],
                "extracted_signatures": ["session_123_signature_1.png", "session_123_signature_2.png"]
            }
        }

def ensure_temp_dir():
    """Create temporary directory if it doesn't exist"""
    temp_dir = Path("temp_signatures")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def cleanup_old_files(temp_dir: Path, max_age_hours: int = 1):
    """Clean up files older than specified hours"""
    current_time = time.time()
    for file_path in temp_dir.glob("*"):
        if current_time - file_path.stat().st_mtime > max_age_hours * 3600:
            file_path.unlink()

def preprocess_image(image):
    """Preprocess the image for signature detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def detect_signature_regions(preprocessed_image):
    """Detect potential signature regions using contour analysis"""
    contours, _ = cv2.findContours(
        preprocessed_image, 
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
            
            if 0.5 < aspect_ratio < 5:
                signature_regions.append((x, y, w, h))
    
    return signature_regions

def extract_signature(image, region, padding: int = 10):
    """Extract signature region with padding"""
    x, y, w, h = region
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)
    
    return image[y_start:y_end, x_start:x_end]

async def process_pdf(pdf_content: bytes, session_id: str):
    """Process PDF content and return detected signatures"""
    temp_dir = ensure_temp_dir()
    cleanup_old_files(temp_dir)
    
    # Save PDF temporarily
    pdf_path = temp_dir / f"{session_id}_document.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    
    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(str(pdf_path))
        
        results = []
        signature_counter = 1
        annotated_pages = []
        extracted_signatures = []
        
        for page_num, image in enumerate(images, 1):
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess and detect signatures
            preprocessed = preprocess_image(image)
            regions = detect_signature_regions(preprocessed)
            
            if regions:
                # Create annotated image
                annotated_image = cv_image.copy()
                
                for region in regions:
                    x, y, w, h = region
                    
                    # Draw rectangle and signature ID
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"#{signature_counter}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, (0, 255, 0), 2)
                    
                    # Extract signature
                    signature_img = extract_signature(cv_image, region)
                    signature_filename = f"{session_id}_signature_{signature_counter}.png"
                    cv2.imwrite(str(temp_dir / signature_filename), signature_img)
                    extracted_signatures.append(signature_filename)
                    
                    # Store signature location
                    results.append(SignatureLocation(
                        page=page_num,
                        signature_id=signature_counter,
                        coordinates={"x": x, "y": y, "width": w, "height": h}
                    ))
                    
                    signature_counter += 1
                
                # Save annotated page
                page_filename = f"{session_id}_page_{page_num}.png"
                cv2.imwrite(str(temp_dir / page_filename), annotated_image)
                annotated_pages.append(page_filename)
        
        return ProcessingResult(
            total_signatures=signature_counter - 1,
            signatures=results,
            annotated_pages=annotated_pages,
            extracted_signatures=extracted_signatures
        )
        
    finally:
        # Clean up temporary PDF
        pdf_path.unlink()

@app.post("/detect-signatures/", 
    response_model=ProcessingResult,
    summary="Detect signatures in a PDF",
    description="""
    Upload a PDF file and detect all signatures within it.
    
    Returns:
    - Total number of signatures found
    - Location and ID of each signature
    - Filenames of annotated pages showing detected signatures
    - Filenames of extracted individual signatures
    
    The annotated pages and extracted signatures can be retrieved using the /files/{filename} endpoint.
    """,
    response_description="Detected signatures and related file information"
)
async def detect_signatures(
    file: UploadFile = File(..., description="PDF file to analyze for signatures")
):
    """
    Upload a PDF and detect signatures within it.
    Returns locations of detected signatures and paths to result files.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a PDF"
        )
    
    content = await file.read()
    session_id = str(uuid.uuid4())
    
    return await process_pdf(content, session_id)

@app.get("/files/{filename}",
    summary="Retrieve processed file",
    description="""
    Retrieve a processed file, which can be either:
    - An annotated page showing detected signatures
    - An extracted signature image
    
    The filename should be obtained from the response of the /detect-signatures/ endpoint.
    """,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "The requested image file"
        },
        404: {
            "description": "File not found"
        }
    }
)
async def get_file(filename: str):
    """Retrieve a processed file (annotated page or extracted signature)"""
    file_path = Path("temp_signatures") / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="File not found"
        )
    
    return FileResponse(
        str(file_path),
        media_type="image/png",
        filename=filename
    )

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Signature Detection API",
        version="1.0.0",
        description="API for detecting and extracting signatures from PDF documents",
        routes=app.routes,
    )

    # Custom documentation extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.on_event("startup")
async def startup_event():
    """Create temporary directory on startup"""
    ensure_temp_dir()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)