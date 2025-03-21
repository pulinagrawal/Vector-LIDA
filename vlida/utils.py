from math import comb
import sys
from matplotlib.pyplot import cla
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
import ollama
import randomname
from lidapy.utils import Node

random_name = lambda: randomname.get_name()

class VectorNode(Node):
    def __init__(self, content, vector, activation, tags=None, links=None):
        super().__init__(content, activation, tags, links)
        self.vector = vector
        self.content = content
        self.activation = activation
        self.tags = tags
        self.links = links
        if links is None:
            self.links = []
        if tags is None:
            self.tags = []
    
    @classmethod
    def combine_nodes(cls, nodes, type):
        match type:
            case 'coalition': _average_embedding_combine(nodes)

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
class EmbeddingFunction:
    def __init__(self, model_name='nomic-embed-text'):
        self.model_name = model_name

    def __call__(self, texts):
        # Tokenize the input text
        return np.array(ollama.embeddings(self.model_name, texts)['embedding'])

class GenerationFunction:
    def __init__(self, model_name='llama2'):
        self.model_name = model_name

    def __call__(self, prompt, system_prompt=''):
        return ollama.generate(self.model_name, prompt, system=system_prompt)


embed = EmbeddingFunction()
generate = GenerationFunction()
get_similarity = lambda x, y: 1-cosine_distance(x, y)
# get_similarity = lambda x, y: 1/(1+np.sqrt(np.linalg.norm(x - y)))