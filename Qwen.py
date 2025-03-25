import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 檢查當前工作目錄
print("目前工作目錄:", os.getcwd())

# 設定模型路徑
MODEL_PATH = "Qwen/Qwen-7B"

def load_model_and_tokenizer(model_path):
    print("正在加載模型和 tokenizer...")
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        trust_remote_code=True  # 修正 ValueError
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("模型與 tokenizer 加載完成！")
    return model, tokenizer

# 嘗試載入模型
try:
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
except Exception as e:
    print("模型加載失敗:", e)
    exit()

# 檢查 sentence-transformers 是否安裝
try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers 模組已安裝！")
except ModuleNotFoundError:
    print("錯誤: 沒有找到 sentence-transformers，請運行 'pip install sentence-transformers' 安裝！")
    exit()

# 檢查 golf_data.txt 是否存在
data_file = os.path.join(os.getcwd(), "golf_data.txt")
if not os.path.exists(data_file):
    print("讀取文件錯誤: 找不到 golf_data.txt，請確認檔案是否存在於", os.getcwd())
    exit()

# 讀取 golf_data.txt
try:
    with open(data_file, "r", encoding="utf-8") as f:
        data = f.read()
    print("成功讀取 golf_data.txt！")
except Exception as e:
    print("讀取文件錯誤:", e)
    exit()

# 簡單測試 tokenizer
sample_text = "高爾夫比賽是如何計分的？"
inputs = tokenizer(sample_text, return_tensors="pt")
print("Tokenized 輸入:", inputs)
