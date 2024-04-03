import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_similarity
import ollama

class Node:
    def __init__(self, vector, text, activation):
        self.vector = vector
        self.text = text
        self.activation = activation

    def __repr__(self) -> str:
        return f"Node(text={self.text}, activation={self.activation})"

def get_most_similar_node(focus_vector, nodes):
    most_similar_node = None
    max_similarity = 0
    for node in nodes:
        similarity = cosine_similarity(focus_vector, node.vector)
        if similarity > max_similarity:
            most_similar_node = node
            max_similarity = similarity
    return most_similar_node, max_similarity

def combine_nodes(nodes, embedding_model):
    # Concatenate text from all nodes
    combined_text = " ".join(node.text for node in nodes)
    
    # Generate a new vector using the embedding model
    combined_vector = embedding_model.encode(combined_text)
    
    # Create a new node with the combined vector and text
    # Assuming a default activation value for the new node, this can be adjusted as needed
    # combined activation is the average of the activations of the nodes being combined
    average_activation = sum(node.activation for node in nodes) / len(nodes)
    combined_node = Node(combined_vector, combined_text, activation=average_activation)
    
    return combined_node

# Assume EmbeddingModel is a placeholder for your actual embedding model with an encode method
class EmbeddingModel:
    def __init__(self, model_name='llama2'):
        self.model_name = model_name

    def encode(self, text):
        # Tokenize the input text
        return ollama.embeddings(self.model_name, text)['embedding']

# Usage
embedding_model = EmbeddingModel()
nodes_to_combine = [Node(np.array([0.1, 0.2, 0.3]), "Hello, world!", 1.0),
                    Node(np.array([0.4, 0.5, 0.6]), "Goodbye, world!", 1.0)]
combined_node = combine_nodes(nodes_to_combine, embedding_model)
get_similarity = lambda x, y: cosine_similarity(x, y)
