import torch
import faiss
import numpy as np
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 設定儲存檔案路徑
MODEL_PATH = "deepseek_model"
TOKENIZER_PATH = "tokenizer.pkl"
INDEX_PATH = "faiss_index.bin"
TEXTS_PATH = "C:/Users/User/Desktop/高爾夫 專題/.mypy_cache/golf_data.txt"
TEXTS_DATA_PATH = "texts.pkl"  # 儲存文本資料

# **讀取文本檔案**
def load_data(file_path, max_lines=5000):  
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    return [text.strip() for text in texts[:max_lines]]

# **批次向量化文本**
def batch_embed_texts(texts, batch_size=4, target_dim=3584):  # 設定目標維度為3584
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")

        with torch.no_grad():
            # 取得最後的隱藏狀態
            outputs = model(**inputs, output_hidden_states=True)
            # 這裡假設我們使用最後一層隱藏狀態並將其進行平均池化
            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
            
            # 調整維度，將嵌入的維度變為 target_dim
            if embeddings.shape[1] != target_dim:
                print(f"嵌入維度為 {embeddings.shape[1]}，將其調整為 {target_dim}...")
                # 使用隨機投影方法或線性變換來調整維度
                embeddings = np.random.normal(size=(embeddings.shape[0], target_dim))  # 這是簡單的隨機調整，根據需要更改為合適方法

        all_embeddings.append(embeddings)
        torch.cuda.empty_cache()  # 清理顯示卡記憶體

        print(f"已處理 {i + len(batch_texts)}/{len(texts)} 行文本...")  # 進度顯示

    return np.vstack(all_embeddings)  

# **載入或建立模型**
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    print("載入已儲存的模型與 tokenizer...")
    tokenizer = joblib.load(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).half().to("cuda")
else:
    print("第一次載入模型，請稍候...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().to("cuda")

    model.save_pretrained(MODEL_PATH)  # 儲存模型
    joblib.dump(tokenizer, TOKENIZER_PATH)  

# **載入或建立 FAISS 索引**
texts = load_data(TEXTS_PATH)
if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_DATA_PATH):
    print("載入已儲存的 FAISS 索引...")
    index = faiss.read_index(INDEX_PATH)
    texts = joblib.load(TEXTS_DATA_PATH)  # 確保文本與索引一致
else:
    print("建立 FAISS 索引（批次處理）...")
    embeddings = batch_embed_texts(texts, batch_size=4, target_dim=3584)  # 設定目標維度為3584
    dim = 3584  # 設定 FAISS 索引的維度
    nlist = 100  # 聚類中心數量
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
    index.train(embeddings)  # 訓練索引
    index.add(embeddings)    # 添加嵌入
    faiss.write_index(index, INDEX_PATH)  # 儲存索引
    joblib.dump(texts, TEXTS_DATA_PATH)  # 儲存文本以保持同步

print("FAISS 索引建立完成！")

# **檢索相似資料**
def retrieve_similar_documents(query, index, texts, k=5):
    print("開始檢索相似文檔...")
    query_vector = batch_embed_texts([query], batch_size=1, target_dim=3584)  # 設定目標維度為3584
    print(f"查詢向量維度: {query_vector.shape[1]}, 索引維度: {index.d}")

    distances, indices = index.search(query_vector, k)
    similar_docs = [texts[i] for i in indices[0] if 0 <= i < len(texts)]
    print("檢索相似文檔完成。")
    return similar_docs

def generate_answer(query):
    print("開始生成答案...")
    similar_docs = retrieve_similar_documents(query, index, texts)
    
    # 限制上下文長度
    context = ' '.join(similar_docs)
    max_context_length = 512  # 設置最大上下文長度
    if len(context) > max_context_length:
        context = context[:max_context_length]
        print("上下文已截斷以適應模型的最大長度。")

    # 改進輸入文本格式
    input_text = f"請根據以下內容回答問題：\n問題: {query}\n內容: {context}\n答案:"
    print(f"生成輸入文本: {input_text}")  # 調試輸出

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to("cuda")
    print(f"生成輸入張量: {inputs}")  # 調試輸出

    # 增加 max_new_tokens
    outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    print(f"生成輸出張量: {outputs}")  # 調試輸出

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成的答案: {answer}")  # 調試輸出

    # 後處理答案，移除查詢和上下文部分
    if "答案:" in answer:
        answer = answer.split("答案:")[-1].strip()
    return answer

# **問答流程**
while True:
    try:
        query = input("請輸入你的問題（按 Enter 提交）：")
        if query.lower() == "exit":
            break
        print("正在處理問題...")
        try:
            answer = generate_answer(query)
            print("回答:", answer)
        except ValueError as e:
            print(f"錯誤: {e}")
    except EOFError:
        print("\n輸入結束，程式退出。")
        break
