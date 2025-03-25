def retrieve_similar_documents(query, k=5):
    query_vector = embed_text(query).numpy()
    distances, indices = index.search(query_vector, k)  # 找到 k 個最相似的資料
    return [texts[i] for i in indices[0]]
