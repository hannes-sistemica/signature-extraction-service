## File Structure

./app/
├── __init__.py
├── main.py
├── models.py
├── database.py
├── config.py 
├── detection/
│   ├── __init__.py
│   ├── preprocessor.py
│   └── extractor.py
├── comparison/
│   ├── __init__.py
│   └── comparator.py
└── utils/
    ├── __init__.py
    ├── file_handler.py
    └── image_utils.py
./requirements.txt

## File Description

### app/main.py
This is the main entry point of the application. It initializes the Flask app and sets up the routes for the API endpoints.

It provides:
1. All necessary imports and dependencies
2. FastAPI app configuration with CORS
3. Main endpoints for signature:
   - Detection
   - Storage
   - Comparison
   - Retrieval
   - Deletion
4. Error handling and parameter validation

The endpoints are RESTful and follow a consistent pattern:
- POST `/signatures/detect` - Detect signatures in documents
- POST `/signatures/store` - Store signatures
- POST `/signatures/compare` - Compare two signatures
- POST `/signatures/find-similar` - Find similar signatures
- GET `/signatures/{id}` - Retrieve stored signature
- DELETE `/signatures/{id}` - Delete stored signature


1. Features:
   - PDF and image support
   - Customizable detection parameters
   - Comparison tuning
   - Similarity search
   - File storage and retrieval

2. Built-in support for:
   - CORS
   - Error handling
   - Parameter validation
   - Async operations
   - File type validation


### app/models.py

The `models.py` defines all necessary Pydantic models with:

1. Parameter Models:
   - `DetectionParams`: Configures signature detection
   - `ComparisonParams`: Controls signature comparison

2. Data Models:
   - `Coordinates`: Signature region location
   - `SignatureLocation`: Signature position in document
   - `ProcessingResult`: Document processing results
   - `ComparisonResult`: Signature comparison results
   - `StoredSignature`: Database signature record

Features:
- Comprehensive field validation
- Detailed descriptions
- Example values for API docs
- Type hints
- Value range constraints
- Nested models

### app/database.py

This file defines the database schema and functions for interacting with it. It includes:

1. Core Database Functions:
   - `init_db()`: Creates tables
   - `get_db()`: Context manager for connections
   - CRUD operations for signatures

2. Tables:
   - `signatures`: Stores signature data and features
   - `detection_results`: Stores detection processing results
   - `comparison_results`: Stores comparison history

3. Key Functions:
   - Signature Management:
     - `store_signature()`
     - `get_signature()`
     - `get_signature_image()`
     - `delete_signature()`
   
   - Comparison:
     - `find_matches()`
     - `compare_features()`
     - `store_comparison_result()`
     - `get_comparison_history()`

4. Features:
   - SQLite with context manager
   - Proper connection handling
   - Pickle for complex data storage
   - Feature comparison logic
   - Error handling
   - Type hints

### app/config.py

This `config.py` provides:

1. Core Configuration:
   - Path settings
   - Database settings
   - Storage settings
   - API settings
   - Processing limits
   - Default parameters

2. Environment Support:
   - Development settings
   - Test settings
   - Production settings
   - Environment variable loading
   - `.env` file support

3. Features:
   - Pydantic validation
   - Automatic directory creation
   - Environment-specific configs
   - Type hints
   - Caching
   - Constants

### app/comparison/comperator.py

The `comparator.py` provides:

1. Main Features:
   - Multiple comparison methods:
     - HOG (Histogram of Oriented Gradients)
     - Contour analysis
     - Density grid comparison
   - Weighted similarity calculation
   - Feature normalization
   - Automatic threshold determination

2. `SignatureComparator` Class:
   - Core comparison logic
   - Configurable parameters
   - Feature normalization
   - Individual feature comparisons
   - Threshold optimization

3. Utility Functions:
   - `compare_signatures`: Convenience function for one-off comparisons
   - `normalize_features`: Feature value normalization
   - `determine_threshold`: Automatic threshold calculation

4. Features:
   - Cosine similarity metrics
   - Array length normalization
   - Weighted scoring
   - Detailed comparison results
   - Type hints
   - Error handling


### app/detection/extractor.py

The `extractor.py` provides:

1. Main Features:
   - Signature feature extraction
   - Multiple feature types:
     - HOG (Histogram of Oriented Gradients)
     - Contour features (area, perimeter, shape descriptors)
     - Density grid features
   - Image preprocessing
   - Signature region extraction

2. `SignatureExtractor` Class:
   - Configurable parameters
   - Robust preprocessing
   - Multiple feature types
   - Error handling
   - Image saving utilities

3. Feature Types:
   - HOG Features:
     - Shape and gradient information
     - Scale-invariant descriptors
   - Contour Features:
     - Area and perimeter
     - Circularity
     - Aspect ratio
     - Extent and solidity
   - Density Features:
     - Grid-based density analysis
     - Configurable grid size

4. Utilities:
   - Image format conversion
   - Error logging
   - Signature saving
   - Convenience functions

### app/detection/preprocessor.py

The `preprocessor.py` provides:

1. Main Features:
   - Document preprocessing
   - Signature detection
   - PDF handling
   - Image processing
   - Result annotation

2. `DocumentPreprocessor` Class:
   - PDF to image conversion
   - Image preprocessing
   - Signature region detection
   - Region validation
   - Signature extraction

3. Processing Steps:
   - Image conversion and normalization
   - Gaussian blur for noise reduction
   - Adaptive thresholding
   - Contour detection
   - Shape analysis
   - Validation checks

4. Utilities:
   - Error handling
   - Logging
   - File handling
   - Format conversion
   - Image annotation

### app/utils/file_handler.py

The `file_handler.py` provides:

1. Main Features:
   - Asynchronous file operations
   - Temporary file management
   - Signature storage
   - File cleanup
   - Metadata handling

2. `FileHandler` Class:
   - File upload handling
   - Size validation
   - Temporary file creation
   - Cleanup routines
   - Metadata management

3. Key Functions:
   - File Operations:
     - `save_uploaded_file`
     - `create_temp_file`
     - `store_signature_file`
   - Management:
     - `cleanup_old_files`
     - `get_file_metadata`
     - `get_signature_path`
   - Validation:
     - `get_file_size`
     - Size limits
     - File type checking

4. Utilities:
   - Error handling
   - Logging
   - Async support
   - Convenience functions
   - Path management

### app/utils/image_utils.py

The `image_utils.py` provides:

1. Format Conversions:
   - `bytes_to_cv2`: Convert bytes to OpenCV format
   - `cv2_to_bytes`: Convert OpenCV to bytes
   - `pil_to_cv2`: Convert PIL to OpenCV
   - `cv2_to_pil`: Convert OpenCV to PIL

2. Image Processing:
   - `resize_image`: Resize with aspect ratio
   - `normalize_image`: Normalize for processing
   - `enhance_signature`: Enhance signature visibility
   - `crop_to_content`: Auto-crop to content
   - `rotate_image`: Rotate with proper bounds
   - `adjust_contrast`: Adjust contrast/brightness

3. Utility Functions:
   - `save_image`: Save with quality settings
   - Proper error handling
   - Logging
   - Type hints
   - Format validation

4. Features:
   - Maintains aspect ratios
   - Handles multiple formats
   - Proper memory management
   - Format conversions
   - Error recovery

