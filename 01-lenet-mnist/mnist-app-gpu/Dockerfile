# One image for both services (trainer and webapp): the CUDA torch wheels are
# several GB, so building them once instead of twice saves disk and build time.
FROM python:3.12-slim

# Where torch comes from. The CUDA 12.6 build matches the host driver (560.94,
# CUDA 12.6). After updating the NVIDIA driver to 580+ the latest CUDA 13 build
# from PyPI can be used instead:
#   TORCH_INDEX_URL=https://pypi.org/simple docker compose build --no-cache
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN pip install torch torchvision --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
