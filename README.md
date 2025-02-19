# Signature Analysis Service

A comprehensive FastAPI-based service for signature detection, extraction, comparison, and management. The service processes both PDF documents and images, identifies signatures using computer vision techniques, and provides capabilities for signature comparison and verification.

## Features

- Signature detection in PDFs and images
- Signature extraction and enhancement
- Signature comparison and similarity matching
- Feature-based signature analysis
- Storage and retrieval of signatures
- REST API with OpenAPI/Swagger documentation
- Docker support

## Prerequisites

The service requires Python 3.11. Check your Python version with:
```bash
python3 --version
```

System dependencies:
```bash
# On macOS
brew install python@3.11 poppler tesseract uv

# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
# Install uv using instructions from: https://github.com/astral-sh/uv
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/signature-service.git
cd signature-service
```

2. Install dependencies using uv:
```bash
make install        # Install production dependencies
make dev-install   # Install development dependencies
```

3. Copy environment template and configure:
```bash
cp example.env .env
```

## Development

### Available Make Commands

- `make help` - Show available commands
- `make install` - Install production dependencies using uv
- `make dev-install` - Install development dependencies
- `make clean` - Clean up temporary files and directories
- `make lint` - Run code quality checks (black, isort, mypy, ruff)
- `make test` - Run tests
- `make run` - Run the FastAPI application locally
- `make docker-build` - Build Docker image
- `make docker-run` - Run Docker container

### Running the Service

```bash
make run  # Runs on http://localhost:8000
```

### Environment Configuration

The service uses environment variables for configuration. Example settings in `.env`:

```ini
# Environment
ENVIRONMENT=dev  # dev, test, or prod

# Paths and Storage
SIGNATURE_BASE_DIR=./app
SIGNATURE_STORAGE_DIR=./storage
SIGNATURE_TEMP_DIR=./temp_signatures
SIGNATURE_DB_PATH=./signatures.db

# Processing Limits
SIGNATURE_MAX_UPLOAD_SIZE=10485760  # 10MB
SIGNATURE_MAX_PDF_PAGES=50
SIGNATURE_PROCESS_TIMEOUT=300

# API Configuration
SIGNATURE_CORS_ORIGINS=["*"]  # Restrict in production
```

## Project Structure
```
app/
├── __init__.py
├── main.py           # FastAPI application and routes
├── models.py         # Pydantic data models
├── database.py       # Database operations
├── config.py         # Configuration settings
├── detection/        # Signature detection
│   ├── preprocessor.py   # Document preprocessing
│   └── extractor.py     # Feature extraction
├── comparison/       # Signature comparison
│   └── comparator.py    # Comparison logic
└── utils/           # Utilities
    ├── file_handler.py  # File operations
    └── image_utils.py   # Image processing
```

## API Usage Examples

### Detect Signatures
```bash
curl -X 'POST' \
  'http://localhost:8000/signatures/detect' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@document.pdf'
```

### Store Signature
```bash
curl -X 'POST' \
  'http://localhost:8000/signatures/store' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@signature.png'
```

### Compare Signatures
```bash
curl -X 'POST' \
  'http://localhost:8000/signatures/compare' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file1=@signature1.png' \
  -F 'file2=@signature2.png'
```

### Find Similar Signatures
```bash
curl -X 'POST' \
  'http://localhost:8000/signatures/find-similar' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@signature.png'
```

## API Documentation

Access the OpenAPI documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.