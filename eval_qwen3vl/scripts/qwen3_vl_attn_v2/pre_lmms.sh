# huggingface 配置
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-}"

# 
# https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec
uv pip install torchcodec==0.10 --index-url=https://download.pytorch.org/whl/cu128