# Signature Detection API Service

A FastAPI-based service that detects and extracts signatures from PDF documents. The service processes PDF files, identifies signature regions using computer vision techniques, and provides annotated results along with extracted signatures.

## Features

- PDF signature detection and extraction
- Annotated PDF pages showing signature locations
- Individual signature extraction as separate images
- Sequential signature numbering across pages
- REST API with OpenAPI/Swagger documentation
- Docker support with multi-architecture builds (amd64/arm64)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Pull and run the service
docker-compose up -d

# View logs
docker-compose logs -f
```

The service will be available at http://localhost:8000

### Using Docker

```bash
# Build the image
docker build -t signature-extraction-service .

# Run the container
docker run -p 8000:8000 signature-extraction-service
```

### Local Development

#### Prerequisites

On macOS, you'll need to install some system dependencies first:

```bash
# Install required system libraries for Pillow and PDF processing
brew install libjpeg zlib poppler tesseract
```

These dependencies are required for:
- `libjpeg`: Image processing with Pillow
- `zlib`: Compression support for Pillow
- `poppler`: PDF to image conversion (used by pdf2image)
- `tesseract`: OCR capabilities (optional, for future use)

#### Installation

```bash
# Install dependencies using uv
make install

# Install development dependencies
make dev-install

# Run the service
make run
```

## API Documentation

Access the OpenAPI documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Usage Examples

### Detect Signatures in a PDF

```bash
curl -X 'POST' \
  'http://localhost:8000/detect-signatures/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/document.pdf'
```

Example Response:
```json
{
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
```

### Retrieve Processed Files

```bash
# Get an annotated page
curl -X 'GET' \
  'http://localhost:8000/files/session_123_page_1.png' \
  -H 'accept: image/png' \
  --output annotated_page.png

# Get an extracted signature
curl -X 'GET' \
  'http://localhost:8000/files/session_123_signature_1.png' \
  -H 'accept: image/png' \
  --output signature.png
```

## Development

### Code Quality

```bash
# Run linting
make lint

# Run tests
make test

# Clean up temporary files
make clean
```

### Available Make Commands

Run `make help` to see all available commands:
- `install`: Install production dependencies
- `dev-install`: Install development dependencies
- `clean`: Clean up temporary files
- `lint`: Run code quality checks
- `test`: Run tests
- `run`: Run the FastAPI application locally
- `docker-build`: Build Docker image
- `docker-run`: Run Docker container

## Configuration

The service stores temporary files in the `temp_signatures` directory, which is:
- Automatically created on startup
- Cleaned up periodically (files older than 1 hour are removed)
- Mounted as a Docker volume when using docker-compose

## GitHub Container Registry

The service is automatically built and published to GitHub Container Registry:
```bash
docker pull ghcr.io/hannes-sistemica/signature-extraction-service:latest
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For support, please open an issue in the GitHub repository.
