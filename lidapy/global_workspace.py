from numpy import broadcast
from lidapy.helpers import get_most_similar_node, create_node 

class Coalition:
    def __init__(self, nodes, attention_codelet_activation):
        self.nodes = nodes
        self.attention_codelet_activation = attention_codelet_activation
        self.activation = self.compute_activation()
        self.create_coalition_node()

    def create_coalition_node(self):
        merged_text = '\n'.join(node.text for node in self.nodes) 
        self.coalition_node = create_node(merged_text, activation=self.compute_activation())

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
        self.create_coalition_node()

    def get_nodes(self):
        return self.nodes   
    
    def is_empty(self):
        return len(self.nodes) == 0

    def __repr__(self) -> str:
        return f"Coalition(nodes={self.nodes}, activation={self.activation})"

class AttentionCodelet:
    def __init__(self, focus_vector=None, focus_tag=None, focus_text=None, activation=1.0):
        self.focus_vector = focus_vector
        self.focus_tag = focus_tag
        self.focus_text = focus_text
        self.activation = activation  # Activation level of the attention codelet

        if focus_vector is not None:
            self.focus = self.focus_vector_coalition
        if focus_text is not None:
            self.focus = self.focus_text_coalition
        if focus_tag is not None:
            self.focus = self.focus_tag_coalition

    def focus_vector_coalition(self, nodes):
        most_similar_node, similiarity = get_most_similar_node(self.focus_vector, nodes)
        return most_similar_node, similiarity 

    def focus_text_coalition(self, nodes):
        pass

    def focus_tag_coalition(self, nodes):
        nodes = []
        for node in nodes:
            if self.focus_tag in node.tags:
                nodes.append(node)
        return max(nodes, key=lambda node: node.activation), 1.0

    def form_coalition(self, csm):
        coalition = Coalition([], self.activation)
        focus_node, similiarity = self.focus(csm.get_all_nodes())
        coalition.add_node(focus_node) 
        coalition.activation *= similiarity
        return coalition

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
        for broadcast_reciever in self.broadcast_receivers:
            broadcast_reciever.recieve_broadcast(winning_coalition)
        return winning_coalition