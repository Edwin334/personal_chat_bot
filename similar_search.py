import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

#loading fancy models
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

#I am using fancy GPU for computing vector search. It's faster but for this amount of data, it does not matter.
#That being said, you can ignore these three following lines(comment out) 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

def split_text(text, max_length=50):
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = []

    current_length = 0
    for token in tokens:
        if current_length + len(token) + 1 > max_length:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += len(token) + 1

    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
    return chunks

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.squeeze().cpu()  # Move the tensor back to CPU


def cosine_similarity(vec1, vec2):
    norm1 = vec1 / np.linalg.norm(vec1)
    norm2 = vec2 / np.linalg.norm(vec2)
    return np.dot(norm1, norm2)

def similar_search(query):
    print("Tokenizer and Model loaded successfully.\n")

    # Load text from stupid_shiwen_sample.txt
    with open('stupid_shiwen_sample.txt', 'r') as file:
        text = file.read()

    chunks = split_text(text)
    embeddings = np.array([embed_text(chunk).numpy() for chunk in chunks])
    query_embedding = embed_text(query).numpy()

    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    indices = np.argsort(similarities)

    # Get the top 5 most similar chunks
    k_nearest_neighbors = indices[-5:][::-1]
    print("\nMagic is happening... Behold: \n")
    print(f"Based on vector search, what's related to the query '{query}'?\n")
    for idx in k_nearest_neighbors:
        print(f"Chunk: {chunks[idx]} - Similarity: {similarities[idx]}")
    return chunks[0]
