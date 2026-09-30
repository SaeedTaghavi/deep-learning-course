FROM python:3.10-slim

WORKDIR /workspace

RUN pip install --no-cache-dir torch torchvision fastapi uvicorn python-multipart jinja2 pillow --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]