FROM nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3

ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace