FROM python:3.12-slim

WORKDIR /app

# Copy source first so pip install can find the package
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the package (needs src/ present for hatchling to find it)
RUN pip install --no-cache-dir .

# Download spaCy English model for NER/entity linking
RUN python -m spacy download en_core_web_sm

COPY migrations/ migrations/

CMD ["uvicorn", "knowledge_service.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
