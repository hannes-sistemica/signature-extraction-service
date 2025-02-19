from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import json
import logging
import sys

from app.models import (
    DetectionParams,
    ComparisonParams,
    ProcessingResult,
    ComparisonResult
)
from app.database import init_db, find_matches, get_signature_path, delete_signature
from app.detection.preprocessor import process_document
from app.detection.extractor import extract_features
from app.comparison.comparator import compare_signatures
from app.utils.file_handler import save_temp_file, cleanup_old_files

import uuid

app = FastAPI(
    title="Signature Analysis Service",
    description="API for signature detection, extraction, storage and comparison",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='INFO:     %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize database and cleanup on startup"""
    logger.info("Starting up signature analysis service")
    init_db()
    cleanup_old_files()
    logger.info("Initialization complete")

@app.post("/signatures/detect",
    response_model=ProcessingResult,
    summary="Detect signatures in a document")
async def detect_signatures(
    file: UploadFile = File(...),
    params: Optional[str] = Form(None)
):
    """
    Detect and extract signatures from a PDF or image file.
    Returns locations and extracted signatures.
    """
    try:
        logger.info(f"Processing document: {file.filename}")
        detection_params = DetectionParams.parse_raw(params) if params else DetectionParams()
        logger.info("Detection parameters parsed successfully")
    except Exception as e:
        logger.error(f"Invalid parameters format: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid parameters format: {str(e)}")
    
    content = await file.read()
    session_id = str(uuid.uuid4())
    
    is_pdf = file.filename.lower().endswith('.pdf')
    is_image = file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
    
    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF or image (PNG, JPG, JPEG, TIFF, BMP)"
        )
    
    return await process_document(content, session_id, is_pdf, detection_params)

@app.post("/signatures/store",
    summary="Store a signature for later comparison")
async def store_signature(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    detection_params: Optional[str] = Form(None),
    comparison_params: Optional[str] = Form(None)
):
    """Store a signature with its features for future comparison"""
    try:
        det_params = DetectionParams.parse_raw(detection_params) if detection_params else DetectionParams()
        comp_params = ComparisonParams.parse_raw(comparison_params) if comparison_params else ComparisonParams()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid parameters format: {str(e)}")
    
    result = await save_temp_file(file, metadata, det_params, comp_params)
    return {"signature_id": result}

@app.post("/signatures/compare",
    response_model=ComparisonResult,
    summary="Compare two signatures")
async def compare_two_signatures(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    params: Optional[str] = Form(None)
):
    """Compare two signature images and return similarity score"""
    try:
        comparison_params = ComparisonParams.parse_raw(params) if params else ComparisonParams()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid parameters format: {str(e)}")
    
    content1 = await file1.read()
    content2 = await file2.read()
    
    features1 = await extract_features(content1, comparison_params)
    features2 = await extract_features(content2, comparison_params)
    
    return compare_signatures(features1, features2, comparison_params)

@app.post("/signatures/find-similar",
    summary="Find similar signatures")
async def find_similar_signatures(
    file: UploadFile = File(...),
    limit: int = Query(5, ge=1, le=100),
    threshold: float = Query(0.7, ge=0, le=1.0),
    params: Optional[str] = Form(None)
):
    """Find similar signatures in the database"""
    try:
        comparison_params = ComparisonParams.parse_raw(params) if params else ComparisonParams(threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid parameters format: {str(e)}")
    
    content = await file.read()
    features = await extract_features(content, comparison_params)
    
    # Get matches from database
    matches = find_matches(features, limit, comparison_params)
    
    return {
        "total_matches": len(matches),
        "matches": matches
    }

@app.get("/signatures/{signature_id}",
    summary="Retrieve stored signature")
async def get_signature(signature_id: str):
    """Retrieve a stored signature by its ID"""
    file_path = get_signature_path(signature_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Signature not found")
    
    return FileResponse(
        str(file_path),
        media_type="image/png",
        filename=f"signature_{signature_id}.png"
    )

@app.delete("/signatures/{signature_id}",
    summary="Delete stored signature")
async def delete_stored_signature(signature_id: str):
    """Delete a stored signature by its ID"""
    success = delete_signature(signature_id)
    if not success:
        raise HTTPException(status_code=404, detail="Signature not found")
    
    return {"message": "Signature deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
