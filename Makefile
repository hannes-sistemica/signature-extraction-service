.PHONY: help install dev-install clean lint test run docker-build docker-run

help: ## Show this help message
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies using uv
	PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 uv pip install -r requirements.txt

dev-install: install ## Install development dependencies
	uv pip install pytest black isort mypy ruff

clean: ## Clean up temporary files and directories
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf temp_signatures/
	rm -rf .venv

lint: ## Run code quality checks
	black app/
	isort app/
	mypy app/
	ruff check app/

test: ## Run tests
	uv pytest

run: ## Run the FastAPI application locally
	PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11 uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build Docker image
	docker build -t signature-extraction-service .

docker-run: ## Run Docker container
	docker run -p 8000:8000 signature-extraction-service
