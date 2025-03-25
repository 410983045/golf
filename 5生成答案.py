from transformers import AutoModelForCausalLM

# 載入模型
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_answer(query):
    similar_docs = retrieve_similar_documents(query)
    input_text = f"Query: {query}\nContext: {' '.join(similar_docs)}"
    
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
