import numpy as np
import logging
from lidapy.utils import get_similarity, combine_nodes, create_node, Node

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
    
    def process(self, nodes):
        associated_nodes = []
        for node in nodes:
            associated_nodes.extend(self.cue(node))
        return associated_nodes

    def store(self, nodes):
        # This should be called upon receiving a broadcast from the Global Workspace
        if isinstance(nodes, Node):
           nodes = [nodes]  
        for node in nodes:
            new_node = Node(vector=node.vector, text=node.text, activation=node.activation, tags=['pam'])
            self.vector_store.add_node(new_node)
            logging.warning(f"PRCP_MEM: Added to vector store: {new_node}")        

    def recieve_broadcast(self, coalition):
        self.store(coalition.coalition_node)
        map(self.store, coalition.get_nodes())
            
    def cue(self, node):
        # Find similar nodes
        similar_nodes = self.vector_store.find_similar_nodes(node.vector, self.threshold)
        logging.warning(f"PRCP_MEM: Similar nodes: {similar_nodes}")        
        # Boost activation of similar nodes
        for similar_node in similar_nodes:
            boost = np.exp(-5 * (1 - similar_node.activation))  # Exponential boost function
            similar_node.activation = min(similar_node.activation + boost, 1.0)  # Cap activation at 1.0
        
        return similar_nodes

# Usage
if __name__ == "__main__":
    pam = PerceptualAssociativeMemory(threshold=0.8)
    node = Node(vector=np.array([0.1, 0.2, 0.3]), text="Hello, world!", activation=1.0)
    pam.process_node(node)
