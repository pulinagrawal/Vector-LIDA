from numpy import broadcast
from lidapy.utils import combine_nodes, get_most_similar_node, create_node 

class Coalition:
    def __init__(self, nodes, attention_codelet):
        self.nodes = nodes
        self.attention_codelet = attention_codelet
        self.activation = self.compute_activation()
        self.coalition_node = combine_nodes(nodes, method='average')

    def compute_activation(self):
        # Average activation of nodes in the coalition
        total_activation = sum(node.activation for node in self.nodes)
        avg_activation = total_activation / len(self.nodes) if self.nodes else 0
        # Weighted by the activation of the attention codelet
        weighted_activation = avg_activation * self.attention_codelet.activation
        return weighted_activation

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)

    def form_coalition(self, focus_nodes, activation=None):
        self.add_nodes(focus_nodes) 
        self.activation = self.compute_activation()
        self.coalition_node = combine_nodes(focus_nodes, method='average')

    def get_nodes(self):
        return self.nodes   
    
    def is_empty(self):
        return len(self.nodes) == 0

    def __repr__(self) -> str:
        return f"Coalition(nodes={self.nodes}, activation={self.activation})"

class GlobalWorkspace:
    def __init__(self, attention_codelets=None, broadcast_receivers=None):
        self.coalitions = []

        self.attention_codelets = []
        if attention_codelets is not None:
            for attention_codelet in attention_codelets:
                self.attention_codelets.append(attention_codelet)

        self.broadcast_receivers = []
        if broadcast_receivers is not None:
            for broadcast_receiver in broadcast_receivers:
                self.broadcast_receivers.append(broadcast_receiver)

    def receive_coalition(self, coalition):
        if coalition is not None:  
            self.coalitions.append(coalition)

    def competition(self):
        if not self.coalitions:
            return None
        # Selecting the coalition with the highest activation
        winning_coalition = max(self.coalitions, key=lambda c: c.activation)
        return winning_coalition

    def decay(self):
        for coalition in self.coalitions:
            coalition.activation *= 0.9

    def run(self, csm):
        self.decay()
        for attention_codelet in self.attention_codelets:
            coalition = attention_codelet.form_coalition(csm)
            self.receive_coalition(coalition)
        winning_coalition = self.competition()
        if winning_coalition is None:  
            return None
        for broadcast_reciever in self.broadcast_receivers:
            broadcast_reciever.recieve_broadcast(winning_coalition)
        return winning_coalition