FROM nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3

ENV HF_HOME=/hf_cache

WORKDIR /workspace