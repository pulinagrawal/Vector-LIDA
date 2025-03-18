import logging
from lidapy.memory import Memory
from lidapy.utils import Node
from vlida.utils import get_similarity, VectorNode as vNode

class VectorStore(Memory):
    def __init__(self, similarity_threshold=0.8):
        self.nodes = []
        self.similarity_threshold = similarity_threshold

    def add_node(self, node):
        self.nodes.append(node)

    def find_associated_nodes(self, node):
        # Find similar nodes
        similar_nodes = self.find_similar_nodes(node.vector, self.similarity_threshold)
        return similar_nodes

    def find_similar_nodes(self, vector, threshold):
        similar_nodes = []
        for node in self.nodes:
            similarity = get_similarity(node.vector, vector)  # cosine similarity
            if similarity >= threshold:
                similar_nodes.append(node)
        return similar_nodes

    def store(self, nodes):
        # This should be called upon receiving a broadcast from the Global Workspace
        if isinstance(nodes, Node):
            nodes = [nodes]
        for node in nodes:
            if not isinstance(node, vNode):
                assert False, f"Node is not a VectorNode: {node}"
            new_node = vNode(vector=node.vector, content=node.content, activation=node.activation, tags=['pam'])
            self.add_node(new_node)

    def learn(self, nodes):
        pass
