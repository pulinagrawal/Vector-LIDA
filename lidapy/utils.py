from math import comb
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
import ollama
import randomname

random_name = lambda: randomname.get_name()

class Node:
    def __init__(self, vector, text, activation, tags=None, links=None):
        self.vector = vector
        self.text = text
        self.activation = activation
        self.tags = tags
        self.links = links
        if links is None:
            self.links = []
        if tags is None:
            self.tags = []

    def __repr__(self) -> str:
        return f"Node(text={self.text}, activation={self.activation})"

def link_nodes(node1, node2):
    node1.links.append(node2)
    node2.links.append(node1)

def get_most_similar_node(focus_vector, nodes):
    most_similar_node = None
    max_similarity = 0
    for node in nodes:
        similarity = get_similarity(focus_vector, node.vector)
        if similarity > max_similarity:
            most_similar_node = node
            max_similarity = similarity
    return most_similar_node, max_similarity

def _text_combine(nodes):
    # Concatenate text from all nodes
    combined_text = " ".join(node.text for node in nodes)
    
    # Generate a new vector using the embedding model
    combined_vector = embed(combined_text)
    
    # Create a new node with the combined vector and text
    # Assuming a default activation value for the new node, this can be adjusted as needed
    # combined activation is the average of the activations of the nodes being combined
    average_activation = sum(node.activation for node in nodes) / len(nodes)
    combined_node = Node(combined_vector, combined_text, activation=average_activation)
    return combined_node
    

average_embedding = lambda nodes: np.mean([node.vector for node in nodes], axis=0)

def _average_embedding_combine(nodes):
    combined_vector = average_embedding(nodes)
    combined_text = "\n and \n".join(node.text for node in nodes)
    average_activation = sum(node.activation for node in nodes) / len(nodes)
    combined_node = Node(combined_vector, combined_text, activation=average_activation)
    return combined_node

def combine_nodes(nodes, method='text'):
    if len(nodes) == 0:
        return None
    if len(nodes) == 1:
        return nodes[0]
    if method == 'text':
        return _text_combine(nodes)
    elif method == 'average':
        return _average_embedding_combine(nodes) 

# Assume EmbeddingModel is a placeholder for your actual embedding model with an encode method
class EmbeddingModel:
    def __init__(self, model_name='llama2'):
        self.model_name = model_name

    def encode(self, text):
        # Tokenize the input text
        return np.array(ollama.embeddings(self.model_name, text)['embedding'])

    def generate(self, prompt):
        return ollama.generate(self.model_name, prompt)


embedding_model = EmbeddingModel()
embed = lambda text: embedding_model.encode(text)
generate = lambda prompt: embedding_model.generate(prompt)
create_node = lambda text, activation=1.0: Node(embed(text), text, activation=activation)
get_similarity = lambda x, y: 1-cosine_distance(x, y)
# get_similarity = lambda x, y: 1/(1+np.sqrt(np.linalg.norm(x - y)))