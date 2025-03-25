from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 使用 8-bit 量化配置
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# 加載標記器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 在 CPU 上加載模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="cpu"  # 修改為使用 CPU
)

print("模型已成功載入！")
