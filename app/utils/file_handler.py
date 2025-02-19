import os
from pathlib import Path
import tempfile
import shutil
import time
from typing import Optional, Union
from fastapi import UploadFile
import logging
import aiofiles
from datetime import datetime, timedelta

from app.config import settings
from app.models import DetectionParams, ComparisonParams

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file operations and temporary storage"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(settings.TEMP_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_uploaded_file(self, 
                               file: UploadFile,
                               prefix: str = "",
                               max_size: Optional[int] = None) -> Path:
        """Save uploaded file to temporary directory"""
        if max_size and await self.get_file_size(file) > max_size:
            raise ValueError(f"File size exceeds maximum limit of {max_size} bytes")

        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{prefix}_{timestamp}_{file.filename}"
        file_path = self.base_dir / unique_filename

        try:
            # Save file using aiofiles for async I/O
            async with aiofiles.open(file_path, 'wb') as out_file:
                while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                    await out_file.write(content)
            
            return file_path

        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {str(e)}")
            if file_path.exists():
                file_path.unlink()
            raise

    async def get_file_size(self, file: UploadFile) -> int:
        """Get size of uploaded file"""
        try:
            # Try to get size from file object
            if hasattr(file, 'size'):
                return file.size
            
            # Otherwise, read and measure content
            file.file.seek(0, 2)  # Seek to end
            size = file.file.tell()
            file.file.seek(0)  # Reset position
            return size

        except Exception as e:
            logger.error(f"Failed to get file size: {str(e)}")
            raise

    def create_temp_file(self,
                        content: Union[str, bytes],
                        suffix: Optional[str] = None,
                        prefix: Optional[str] = None) -> Path:
        """Create temporary file with content"""
        try:
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(suffix=suffix,
                                           prefix=prefix,
                                           dir=self.base_dir)
            
            # Write content
            mode = 'wb' if isinstance(content, bytes) else 'w'
            with os.fdopen(fd, mode) as temp_file:
                temp_file.write(content)
            
            return Path(temp_path)

        except Exception as e:
            logger.error(f"Failed to create temporary file: {str(e)}")
            raise

    def cleanup_old_files(self, max_age: timedelta = timedelta(hours=1)) -> int:
        """Clean up files older than specified age"""
        count = 0
        current_time = datetime.now()
        
        try:
            for file_path in self.base_dir.glob("*"):
                if file_path.is_file():
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if current_time - file_age > max_age:
                        file_path.unlink()
                        count += 1
            
            return count

        except Exception as e:
            logger.error(f"Error during file cleanup: {str(e)}")
            raise

    def get_file_metadata(self, file_path: Path) -> dict:
        """Get file metadata"""
        try:
            stat = file_path.stat()
            return {
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "extension": file_path.suffix.lower(),
                "name": file_path.name
            }

        except Exception as e:
            logger.error(f"Failed to get file metadata: {str(e)}")
            raise

    async def store_signature_file(self,
                                 file: UploadFile,
                                 signature_id: str,
                                 metadata: Optional[str] = None,
                                 detection_params: Optional[DetectionParams] = None,
                                 comparison_params: Optional[ComparisonParams] = None) -> Path:
        """Store signature file with metadata"""
        try:
            # Create directory structure
            signature_dir = self.base_dir / "signatures" / signature_id
            signature_dir.mkdir(parents=True, exist_ok=True)
            
            # Save signature file
            file_path = signature_dir / f"signature{Path(file.filename).suffix}"
            async with aiofiles.open(file_path, 'wb') as out_file:
                while content := await file.read(1024 * 1024):
                    await out_file.write(content)
            
            # Save metadata if provided
            if any([metadata, detection_params, comparison_params]):
                meta_data = {
                    "metadata": metadata,
                    "detection_params": detection_params.dict() if detection_params else None,
                    "comparison_params": comparison_params.dict() if comparison_params else None,
                    "original_filename": file.filename,
                    "timestamp": datetime.now().isoformat()
                }
                
                meta_path = signature_dir / "metadata.json"
                async with aiofiles.open(meta_path, 'w') as meta_file:
                    await meta_file.write(json.dumps(meta_data, indent=2))
            
            return file_path

        except Exception as e:
            logger.error(f"Failed to store signature file: {str(e)}")
            if 'signature_dir' in locals() and signature_dir.exists():
                shutil.rmtree(signature_dir)
            raise

    def get_signature_path(self, signature_id: str) -> Optional[Path]:
        """Get path of stored signature file"""
        signature_dir = self.base_dir / "signatures" / signature_id
        
        if not signature_dir.exists():
            return None
        
        # Find signature file
        signature_files = list(signature_dir.glob("signature.*"))
        return signature_files[0] if signature_files else None

    def delete_signature(self, signature_id: str) -> bool:
        """Delete stored signature and its metadata"""
        signature_dir = self.base_dir / "signatures" / signature_id
        
        if not signature_dir.exists():
            return False
        
        try:
            shutil.rmtree(signature_dir)
            return True
        except Exception as e:
            logger.error(f"Failed to delete signature {signature_id}: {str(e)}")
            return False

# Global file handler instance
file_handler = FileHandler()

# Convenience functions
async def save_temp_file(
    file: UploadFile,
    metadata: Optional[str] = None,
    detection_params: Optional[DetectionParams] = None,
    comparison_params: Optional[ComparisonParams] = None
) -> str:
    """Save temporary file and return its ID"""
    signature_id = str(uuid.uuid4())
    await file_handler.store_signature_file(
        file,
        signature_id,
        metadata,
        detection_params,
        comparison_params
    )
    return signature_id

def cleanup_old_files(max_age_hours: int = 1) -> int:
    """Clean up old temporary files"""
    return file_handler.cleanup_old_files(timedelta(hours=max_age_hours))

def get_signature_path(signature_id: str) -> Optional[Path]:
    """Get path to stored signature"""
    return file_handler.get_signature_path(signature_id)

def delete_stored_signature(signature_id: str) -> bool:
    """Delete stored signature"""
    return file_handler.delete_signature(signature_id)