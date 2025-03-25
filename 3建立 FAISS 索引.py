import faiss
import numpy as np

# 假設你有一個文本資料列表
texts = ["你的文本資料1", "你的文本資料2", ...]
embeddings = [embed_text(text).numpy() for text in texts]

# 建立 FAISS 索引
dim = embeddings[0].shape[1]  # 向量維度
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))  # 將向量添加到索引中
