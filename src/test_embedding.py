import ollama

response = ollama.embeddings(
    model='embeddinggemma:latest',
    prompt='Triệu chứng của bệnh tiểu đường là gì?'
)
print(f"Embedding dimension: {len(response['embedding'])}")
print(f"First 10 values: {response['embedding'][:10]}")