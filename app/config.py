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
    DEFAULT_IMAGE_SIZE: tuple = (1654, 2340)  # A4 size at 200 DPI
    DEFAULT_GRID_SIZE: int = 5
    
    # Detection defaults
    MIN_AREA: int = 500
    COMPLEXITY_THRESHOLD: float = 10.0  # Adjusted for better detection
    GAUSSIAN_BLUR_KERNEL: int = 5
    ADAPTIVE_BLOCK_SIZE: int = 11
    ADAPTIVE_C: int = 2
    CONTOUR_APPROX_FACTOR: float = 0.02
    PADDING: int = 10
    ASPECT_RATIO_MIN: float = 0.2  # More lenient
    ASPECT_RATIO_MAX: float = 8.0  # More lenient
    DENSITY_MIN: float = 0.05
    DENSITY_MAX: float = 0.85
    SOLIDITY_MIN: float = 0.1
    SOLIDITY_MAX: float = 0.98
    MAX_DETECTION_PASSES: int = 4  # Maximum number of detection passes per page
    
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
