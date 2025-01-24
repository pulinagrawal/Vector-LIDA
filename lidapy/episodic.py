from lidapy.utils import get_most_similar_node, create_node

class EpisodicMemory:
    def __init__(self):
        self.vector_store = []

    def store(self, incoming_node):
        node = create_node(incoming_node.text, activation=incoming_node.activation)
        node.tags.append('episodic')
        self.vector_store.append(node)

    def cue(self, vector):
        similar_node, similarity = get_most_similar_node(vector, self.vector_store)
        return similar_node
