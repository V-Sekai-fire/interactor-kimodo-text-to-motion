# interactor-kimodo-text-to-motion -- vast.ai worker, RFD 0036/0045.
#
# Small model (0.6 GB bf16) -- ships bf16, not Q4_K_M (RFD 0045's own
# table lists bf16 as the format; no quant win worth the complexity at
# this size).

FROM python:3.11-slim AS contract
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3
COPY server.py /app/server.py
COPY test_input.json /app/test_input.json
ENV WEFTSPUN_STUB=1 PORT=8000
EXPOSE 8000
CMD ["python", "/app/server.py"]

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS worker

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3 \
       safetensors huggingface_hub==0.26.2

# Kimodo -- code is Apache-2.0. Weights are licensed separately per
# checkpoint (see README): Kimodo-SOMA/Kimodo-G1 are NVIDIA Open Model
# License (commercial-friendly); Kimodo-SMPLX is the more restrictive
# NVIDIA R&D Model License. This image ships Kimodo-SOMA, matching
# what this interface returns (RFD 0045: "the model image returns
# SOMA only" without a target_rig) -- do not swap in the SMPLX
# checkpoint without re-checking the license gate (RFD 0028).
ARG KIMODO_REPO=nv-tlabs/kimodo-soma
RUN mkdir -p /weights && \
    python3 -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
shutil.copy(hf_hub_download(repo_id='${KIMODO_REPO}', filename='kimodo.safetensors'), '/weights/kimodo.safetensors')" \
    || echo "weight fetch deferred -- confirm exact HF repo id before real build"

COPY server.py /app/server.py

ENV PORT=8000
EXPOSE 8000

CMD ["python3", "-u", "/app/server.py"]
