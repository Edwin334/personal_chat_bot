import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Set device to Metal GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model and tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model.to(device)

def split_text(text, max_length=20):
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

def main():
    print("Tokenizer and Model loaded successfully.\n")

    #some ramdom example.
    text = (f"ITEM 2. MANAGEMENT’S DISCUSSION AND"
            f"ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONSThe following discussion"
            f"and analysis should be read in conjunction with the consolidated financial statements"
            f"and the related notes included elsewhere in this Quarterly Report on Form 10-Q.OverviewOur"
            f"mission is to accelerate the world’s transition to sustainable energy. We design, develop, manufacture, lease and sell high-performance fully electric vehicles, solar energy generation systems and energy storage products. We also offer maintenance, installation, operation, charging, insurance, financial and other services related to our products. Additionally, we are increasingly focused on products and services based on AI, robotics and automation. During the three and six months ended June 30, 2023, we recognized total revenues of $24.93 billion and $48.26 billion, respectively. We continue to ramp production, build new manufacturing capacity, invest in research and development and expand our operations to enable increased deliveries and deployments of our products and further revenue growth.During the three and six months ended June 30, 2023, our net income attributable to common stockholders was $2.70 billion and $5.22 billion, respectively. We continue to focus on improving our profitability through production and operational efficiencies.")

    chunks = split_text(text)
    embeddings = np.array([embed_text(chunk).numpy() for chunk in chunks])

    query = "total revenue"
    query_embedding = embed_text(query).numpy()

    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    indices = np.argsort(similarities)

    # Get the top 5 most similar chunks
    k_nearest_neighbors = indices[-5:][::-1]
    print("\nMagic is happening... Behold: \n")
    print(f"Based on vector search, what's related to the query '{query}'?\n")
    for idx in k_nearest_neighbors:
        print(f"Chunk: {chunks[idx]} - Similarity: {similarities[idx]}")

if __name__ == "__main__":
    main()
