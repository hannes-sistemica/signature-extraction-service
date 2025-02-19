from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class DetectionParams(BaseModel):
    """Parameters for signature detection fine-tuning"""
    min_area: int = Field(
        default=500,
        description="Minimum contour area to consider as signature",
        ge=100
    )
    complexity_threshold: float = Field(
        default=20.0,
        description="Threshold for contour complexity",
        ge=1.0
    )
    gaussian_blur_kernel: Tuple[int, int] = Field(
        default=(5, 5),
        description="Kernel size for Gaussian blur"
    )
    gaussian_blur_sigma: int = Field(
        default=0,
        description="Sigma for Gaussian blur",
        ge=0
    )
    adaptive_block_size: int = Field(
        default=11,
        description="Block size for adaptive threshold",
        ge=3,
        le=99
    )
    adaptive_c: int = Field(
        default=2,
        description="C parameter for adaptive threshold",
        ge=0
    )
    contour_approx_factor: float = Field(
        default=0.02,
        description="Factor for contour approximation",
        gt=0.0,
        le=1.0
    )
    padding: int = Field(
        default=10,
        description="Padding around detected signatures",
        ge=0
    )
    aspect_ratio_min: float = Field(
        default=0.5,
        description="Minimum width/height ratio",
        gt=0.0
    )
    aspect_ratio_max: float = Field(
        default=5.0,
        description="Maximum width/height ratio",
        gt=1.0
    )

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "min_area": 500,
                "complexity_threshold": 20.0,
                "gaussian_blur_kernel": [5, 5],
                "gaussian_blur_sigma": 0,
                "adaptive_block_size": 11,
                "adaptive_c": 2,
                "contour_approx_factor": 0.02,
                "padding": 10,
                "aspect_ratio_min": 0.5,
                "aspect_ratio_max": 5.0
            }
        }

class ComparisonParams(BaseModel):
    """Parameters for signature comparison"""
    threshold: float = Field(
        default=0.7,
        description="Minimum similarity score to consider a match",
        ge=0.0,
        le=1.0
    )
    grid_size: int = Field(
        default=5,
        description="Size of grid for density features (N x N)",
        ge=3,
        le=20
    )
    hog_weight: float = Field(
        default=1.0,
        description="Weight for HOG features in comparison",
        ge=0.0
    )
    contour_weight: float = Field(
        default=1.0,
        description="Weight for contour features in comparison",
        ge=0.0
    )
    density_weight: float = Field(
        default=1.0,
        description="Weight for density features in comparison",
        ge=0.0
    )
    binary_threshold: int = Field(
        default=127,
        description="Threshold for binary conversion",
        ge=0,
        le=255
    )

    class Config:
        schema_extra = {
            "example": {
                "threshold": 0.7,
                "grid_size": 5,
                "hog_weight": 1.0,
                "contour_weight": 1.0,
                "density_weight": 1.0,
                "binary_threshold": 127
            }
        }

class Coordinates(BaseModel):
    """Coordinates and dimensions of a detected signature"""
    x: int = Field(..., description="X coordinate of signature region")
    y: int = Field(..., description="Y coordinate of signature region")
    width: int = Field(..., description="Width of signature region")
    height: int = Field(..., description="Height of signature region")

class SignatureLocation(BaseModel):
    """Location of a detected signature within a document"""
    page: int = Field(..., description="Page number in document")
    signature_id: int = Field(..., description="Unique identifier for this signature")
    coordinates: Coordinates

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
    """Results of processing a document for signatures"""
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
                    }
                ],
                "annotated_pages": ["session_123_page_1.png"],
                "extracted_signatures": ["session_123_signature_1.png"]
            }
        }

class ComparisonResult(BaseModel):
    """Results of comparing two signatures"""
    similarity_score: float = Field(
        ...,
        description="Overall similarity score",
        ge=0.0,
        le=1.0
    )
    is_match: bool = Field(
        ...,
        description="Whether the signatures are considered a match"
    )
    details: Dict[str, float] = Field(
        ...,
        description="Detailed similarity scores for each feature type"
    )

    class Config:
        schema_extra = {
            "example": {
                "similarity_score": 0.85,
                "is_match": True,
                "details": {
                    "hog": 0.87,
                    "contour": 0.82,
                    "density": 0.86
                }
            }
        }

class StoredSignature(BaseModel):
    """Information about a stored signature"""
    id: str
    filename: str
    metadata: Optional[str]
    features: Dict[str, List[float]]
    created_at: datetime

    class Config:
        schema_extra = {
            "example": {
                "id": "abc123",
                "filename": "signature.png",
                "metadata": "John Doe's signature",
                "features": {
                    "hog": [0.1, 0.2, 0.3],
                    "contour": [0.4, 0.5],
                    "density": [0.6, 0.7, 0.8]
                },
                "created_at": "2024-02-19T10:30:00"
            }
        }