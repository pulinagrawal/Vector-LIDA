import torch
from transformers import AutoTokenizer, AutoModel
import ollama

# Load pre-trained model and tokenizer
model_name = "llama2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_embedding_ollama(text):
    return ollama.embeddings(model_name, text)

def get_embedding_lhs(text):
    """Get the embedding of a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling to get a fixed-size vector representation
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding

def get_embedding_pool(text):
    """Get the embedding of a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling to get a fixed-size vector representation
        embedding = outputs.pooler_output
    return embedding

# Define similar and dissimilar texts
similar_texts = ["I love programming.", "Coding is my passion."]
dissimilar_texts = ["I love programming.", "The weather is sunny."]

# Get embeddings
similar_embeddings = [get_embedding_lhs(text) for text in similar_texts]
dissimilar_embeddings = [get_embedding_lhs(text) for text in dissimilar_texts]

# Calculate cosine similarities
similar_similarity = torch.cosine_similarity(similar_embeddings[0], similar_embeddings[1])
dissimilar_similarity = torch.cosine_similarity(dissimilar_embeddings[0], dissimilar_embeddings[1])

print(f"Cosine similarity between similar texts: {similar_similarity:.4f}")
print(f"Cosine similarity between dissimilar texts: {dissimilar_similarity:.4f}")

