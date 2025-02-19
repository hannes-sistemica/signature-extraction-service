from pydantic import BaseSettings
from pathlib import Path
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    STORAGE_DIR: Path = Path("storage")
    TEMP_DIR: Path = Path("temp_signatures")
    
    # Database settings
    DB_PATH: Path = Path("signatures.db")
    DB_POOL_SIZE: int = 5
    DB_POOL_TIMEOUT: int = 30
    
    # File storage settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    MAX_FILE_AGE_HOURS: int = 24
    
    # Processing settings
    MAX_PDF_PAGES: int = 50
    MAX_SIGNATURES_PER_PAGE: int = 10
    PROCESS_TIMEOUT: int = 300  # 5 minutes
    
    # API settings
    API_TITLE: str = "Signature Analysis Service"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for signature detection, extraction, storage and comparison"
    CORS_ORIGINS: List[str] = ["*"]
    
    # Feature extraction defaults
    DEFAULT_IMAGE_SIZE: tuple = (300, 150)
    DEFAULT_GRID_SIZE: int = 5
    
    # Detection defaults
    DETECTION_DEFAULTS: Dict[str, Any] = {
        "min_area": 500,
        "complexity_threshold": 20.0,
        "gaussian_blur_kernel": (5, 5),
        "gaussian_blur_sigma": 0,
        "adaptive_block_size": 11,
        "adaptive_c": 2,
        "contour_approx_factor": 0.02,
        "padding": 10,
        "aspect_ratio_min": 0.5,
        "aspect_ratio_max": 5.0
    }
    
    # Comparison defaults
    COMPARISON_DEFAULTS: Dict[str, Any] = {
        "threshold": 0.7,
        "grid_size": 5,
        "hog_weight": 1.0,
        "contour_weight": 1.0,
        "density_weight": 1.0,
        "binary_threshold": 127
    }

    class Config:
        env_prefix = "SIGNATURE_"
        env_file = ".env"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name == "BASE_DIR":
                return Path(raw_val)
            elif field_name in ["STORAGE_DIR", "TEMP_DIR", "DB_PATH"]:
                return Path(raw_val)
            return cls.json_loads(raw_val)

# Initialize settings
settings = Settings()

# Create necessary directories
settings.TEMP_DIR.mkdir(exist_ok=True, parents=True)
settings.STORAGE_DIR.mkdir(exist_ok=True, parents=True)
settings.DB_PATH.parent.mkdir(exist_ok=True, parents=True)