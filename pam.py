import numpy as np
import logging
from helpers import get_similarity, combine_nodes

class Node:
    def __init__(self, vector, text, activation):
        self.vector = vector
        self.text = text
        self.activation = activation

class VectorStore:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def find_similar_nodes(self, vector, threshold):
        similar_nodes = []
        for node in self.nodes:
            similarity = get_similarity(node.vector, vector)  # cosine similarity
            if similarity >= threshold:
                similar_nodes.append(node)
        return similar_nodes

class PerceptualAssociativeMemory:
    def __init__(self, threshold=0.8):
        self.vector_store = VectorStore()
        self.threshold = threshold
    
    def process_node(self, node):
        # Find similar nodes
        similar_nodes = self.vector_store.find_similar_nodes(node.vector, self.threshold)
        logging.warning(f"PRCP_MEM: Similar nodes: {similar_nodes}")        
        # Boost activation of similar nodes
        for similar_node in similar_nodes:
            boost = np.exp(-5 * (1 - similar_node.activation))  # Exponential boost function
            similar_node.activation = min(similar_node.activation + boost, 1.0)  # Cap activation at 1.0
        
        # Add the new node to the vector store if no similar node is found
        if not len(similar_nodes):
            self.store(node)

    def store(self, node):
        self.vector_store.add_node(node)
        logging.warning(f"PRCP_MEM: Added to vector store: {node}")        
            
    def cue(self, vector):
        similar_nodes = self.vector_store.find_similar_nodes(vector, self.threshold)
        combined_node = combine_nodes(similar_nodes)
        return combined_node

# Usage
pam = PerceptualAssociativeMemory(threshold=0.8)
node = Node(vector=np.array([0.1, 0.2, 0.3]), text="Hello, world!", activation=1.0)
pam.process_node(node)
