import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 設置設備為 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加載模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=torch.float16).to(device)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# 定義要提問的文本
question = "高爾夫比賽的規則是什麼？"

# 將問題編碼為模型可接受的格式
inputs = tokenizer.encode(question, return_tensors="pt").to(device)

# 使用模型生成回答
with torch.no_grad():
    outputs = model.generate(inputs, max_length=100)  # max_length 可以根據需要調整

# 解碼生成的回答
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("回答:", answer)
