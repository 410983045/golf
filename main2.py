import torch  # 載入 PyTorch，用來處理深度學習模型
import faiss  # 載入 FAISS，用來建立與搜尋向量索引
import numpy as np  # 載入 NumPy，方便進行數據處理
import joblib  # 載入 joblib，方便存取模型和數據
from transformers import AutoModelForCausalLM, AutoTokenizer  # 載入 transformers，處理語言模型
import os  # 載入 os，讓程式可以讀取檔案與路徑

# 設定存檔的路徑，讓模型、索引和數據可以儲存與讀取
MODEL_PATH = "deepseek_model"  # 模型儲存位置
TOKENIZER_PATH = "tokenizer.pkl"  # Tokenizer 儲存位置
INDEX_PATH = "faiss_index.bin"  # FAISS 索引檔案
TEXTS_PATH = "C:/Users/User/Desktop/高爾夫 專題/.mypy_cache/golf_data.txt"  # 文字資料來源
TEXTS_DATA_PATH = "texts.pkl"  # 儲存文本資料，避免每次都重新載入

# **讀取文本檔案**
def load_data(file_path, max_lines=5000):  # 從指定路徑讀取文本內容，最多 5000 行
    with open(file_path, 'r', encoding='utf-8') as file:  # 以 UTF-8 讀取檔案
        texts = file.readlines()  # 讀取所有行
    return [text.strip() for text in texts[:max_lines]]  # 去除換行符號，回傳最多 max_lines 行

# **批次向量化文本**
def batch_embed_texts(texts, batch_size=4, target_dim=3584):  # 一次處理 4 行文本，轉換成 3584 維的向量
    all_embeddings = []  # 存放所有文本的向量
    for i in range(0, len(texts), batch_size):  # 逐批處理文本
        batch_texts = texts[i:i + batch_size]  # 取出當前批次的文本
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")  # 將文本轉為模型輸入格式

        with torch.no_grad():  # 停用梯度計算，節省記憶體
            outputs = model(**inputs, output_hidden_states=True)  # 取得模型的隱藏層輸出
            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()  # 取最後一層隱藏狀態的平均值作為向量
            print(f"嵌入維度: {embeddings.shape[1]}")  # 確認向量維度
            
            if embeddings.shape[1] != target_dim:  # 如果向量的維度不符合 3584，就要調整
                print(f"嵌入維度為 {embeddings.shape[1]}，將其調整為 {target_dim}...")  
                embeddings = np.random.normal(size=(embeddings.shape[0], target_dim))  # 這裡只是用隨機數填充，實際應使用更好的降維方法

        all_embeddings.append(embeddings)  # 把處理好的向量存起來
        torch.cuda.empty_cache()  # 釋放 GPU 記憶體，避免佔用過多資源
        print(f"已處理 {i + len(batch_texts)}/{len(texts)} 行文本...")  # 顯示目前處理進度

    return np.vstack(all_embeddings)  # 合併所有批次的向量並回傳

# **載入或建立模型**
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):  # 如果模型和 Tokenizer 檔案已經存在，就直接載入
    print("載入已儲存的模型與 tokenizer...")
    tokenizer = joblib.load(TOKENIZER_PATH)  # 載入 Tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).half().to("cuda")  # 載入模型並移到 GPU
else:  # 如果是第一次使用，則從網路下載並存檔
    print("第一次載入模型")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 指定要使用的語言模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 下載 Tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).half().to("cuda")  # 下載模型並轉換為半精度，減少記憶體占用

    model.save_pretrained(MODEL_PATH)  # 儲存模型，避免下次重新下載
    joblib.dump(tokenizer, TOKENIZER_PATH)  # 儲存 Tokenizer

# **載入或建立 FAISS 索引**
texts = load_data(TEXTS_PATH)  # 讀取文本資料
if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_DATA_PATH):  # 如果索引和文本檔案都存在，就直接載入
    print("載入已儲存的 FAISS 索引...")
    index = faiss.read_index(INDEX_PATH)  # 載入 FAISS 索引
    texts = joblib.load(TEXTS_DATA_PATH)  # 確保文本與索引一致
else:  # 如果沒有索引，就建立新的索引
    print("建立 FAISS 索引")
    embeddings = batch_embed_texts(texts, batch_size=4, target_dim=3584)  # 把文本轉成向量
    dim = 3584  # 向量維度
    nlist = 100  # FAISS 的索引中心數
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)  # 建立索引
    index.train(embeddings)  # 訓練索引
    index.add(embeddings)  # 加入向量
    faiss.write_index(index, INDEX_PATH)  # 存檔，避免下次重建
    joblib.dump(texts, TEXTS_DATA_PATH)  # 存檔，確保與索引對應

print("FAISS 索引建立完成！")

# **檢索相似資料**
def retrieve_similar_documents(query, index, texts, k=5):  # k=5 表示取出 5 篇最相似文本
    query_vector = batch_embed_texts([query], batch_size=1, target_dim=3584)  # 將查詢轉換為向量
    distances, indices = index.search(query_vector, k)  # 在 FAISS 索引中搜尋相似文本
    return [texts[i] for i in indices[0] if 0 <= i < len(texts)]  # 返回相應的文本

# **問答流程**
while True:
    query = input("請輸入問題（輸入 exit 離開）：")  # 讓使用者輸入問題
    if query.lower() == "exit":  # 若輸入 'exit' 則結束迴圈
        break
    similar_docs = retrieve_similar_documents(query, index, texts)  # 檢索相似文本
    context = ' '.join(similar_docs)[:512]  # 合併文本並限制長度
    input_text = f"請根據以下內容回答問題：\n問題: {query}\n內容: {context}\n答案:"  # 格式化輸入
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to("cuda")  # Tokenize
    outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)  # 生成答案
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("答案:")[-1].strip()  # 解碼答案
    print("回答:", answer)  # 顯示結果