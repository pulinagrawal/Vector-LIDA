from collections import deque
from helpers import combine_nodes, Node, get_most_similar_node

class CurrentSituationalModel:
    def __init__(self, max_size, embedding_model):
        self.nodes_queue = deque(maxlen=max_size)  # Fixed-size queue to store nodes
        self.embedding_model = embedding_model # Embedding model to combine nodes

    def add_node(self, node):
        self.nodes_queue.append(node)  # Adds a new node to the right end of the queue

    def cue_pam(self, vector, pam, threshold):
        # Assume pam is an instance of PerceptualAssociativeMemory
        similar_nodes = pam.vector_store.find_similar_nodes(vector, threshold)
        combined_node = combine_nodes(similar_nodes, self.embedding_model)
        return combined_node
        
    def get_all_nodes(self):
        return list(self.nodes_queue)

class Coalition:
    def __init__(self, nodes, attention_codelet_activation):
        self.nodes = nodes
        self.attention_codelet_activation = attention_codelet_activation
        self.activation = self.compute_activation()

    def compute_activation(self):
        # Average activation of nodes in the coalition
        total_activation = sum(node.activation for node in self.nodes)
        avg_activation = total_activation / len(self.nodes) if self.nodes else 0
        # Weighted by the activation of the attention codelet
        weighted_activation = avg_activation * self.attention_codelet_activation
        return weighted_activation

    def add_node(self, node):
        self.nodes.append(node)
        self.activation = self.compute_activation()

    def get_nodes(self):
        return self.nodes   
    
    def is_empty(self):
        return len(self.nodes) == 0

    def __repr__(self) -> str:
        return f"Coalition(nodes={self.nodes}, activation={self.activation})"

class AttentionCodelet:
    def __init__(self, focus_vector, activation=1.0):
        self.focus_vector = focus_vector
        self.activation = activation  # Activation level of the attention codelet

    def form_coalition(self, csm):
        coalition = Coalition([], self.activation)
        most_similar_node, similiarity = get_most_similar_node(self.focus_vector, csm.get_all_nodes())
        most_similar_node.activation *= similiarity
        coalition.add_node(most_similar_node)
        return coalition

class GlobalWorkspace:
    def __init__(self):
        self.coalitions = []

    def receive_coalition(self, coalition):
        self.coalitions.append(coalition)

    def competition(self):
        if not self.coalitions:
            return None
        # Selecting the coalition with the highest activation
        winning_coalition = max(self.coalitions, key=lambda c: c.activation)
        return winning_coalition
