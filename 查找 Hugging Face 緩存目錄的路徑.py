import os

# 獲取 Hugging Face 緩存目錄
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print("Hugging Face cache directory:", cache_dir)
