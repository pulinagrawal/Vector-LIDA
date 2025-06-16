import numpy as np

from pathlib import Path
from PIL import Image
from sympy import deg
from lidapy.utils import Node
from lidapy.memory import Memory
from film_agent.clip_utils import clip_image_encoder
from film_agent.utils import compute_average_embedding

class DefaultPAMMemory(Memory):
    def __init__(self):
        self.nodes = set()

    def store(self, nodes):
        for node in nodes:
            for existing_node in self.nodes:
                if existing_node.content == node.content:
                    return
            self.nodes.add(node)
    
    def find_associated_nodes(self, node):
        self.learn([node])
        associated_nodes = []
        associated_nodes.extend([n for n in node.links])
        return associated_nodes

    def learn(self, nodes):
        for node in nodes:
            self.store(node)

class MobileCLIPPAMMemory(DefaultPAMMemory):
    def __init__(self, bootstrap_nodes=None):
        super().__init__()
        if bootstrap_nodes is not None:
            self.store(bootstrap_nodes)

    def find_associated_nodes(self, node):
        """
        Find nodes associated with the given node based on similarity of features.
        This method overrides the default behavior to use CLIP-based similarity.
        """
        associated_nodes = []
        
        most_similar = None
        best_similarity = 0
        for stored_node in self.nodes:
            if stored_node != node:
                similarity = np.dot(node.features, stored_node.features) / (np.linalg.norm(node.features) * np.linalg.norm(stored_node.features))
                if similarity > best_similarity:
                    best_similarity = similarity
                    most_similar = stored_node
        if most_similar is not None:
            associated_nodes.append(most_similar)
        return associated_nodes

    def learn(self, nodes):
        for node in nodes:
            # Find associated nodes based on CLIP similarity
            associated_nodes = self.find_associated_nodes(node)
            
            # Link the current node to its associated nodes
            for associated_node in associated_nodes:
                associated_node.features = compute_average_embedding(node.features, prev_embedding=associated_node.features)