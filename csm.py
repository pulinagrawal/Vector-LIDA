from collections import deque
from helpers import combine_nodes, Node, get_most_similar_node, embed, create_node, get_similarity

class StructureBuildingCodelet:
    def __init__(self, interests = [], activation=1.0):
        self.activation = activation
        self.interests = interests

    def run(self, csm):
        pass


class CurrentSituationalModel:
    def __init__(self, max_size, sbcs=None, memories=None):
        self.nodes = [] 
        self.ccq = deque(maxlen=max_size)  # Conscious Contents Queue
        self.sbcs = sbcs
        self.memories = memories

    def run_structure_building_codelets(self):
        for sbc in self.sbcs:
            new_structures = sbc.run(self)
            for structure in new_structures:
                self.add_node(structure)

    def add_node(self, node):
        self.nodes.append(node)  # Adds a new node to the right end of the queue

    def decay(self, nodes):
        for node in nodes:
            node.activation *= 0.9

    def run(self, node):
        self.decay(self.nodes)
        self.add_node(node)
        self.run_structure_building_codelets()

    def receive_broadcast(self, coalition):
        node = coalition.coalition_node
        self.ccq.appendleft(node)
        self.cue_memories(node)

    def cue_memories(self, node):
        for memory in self.memories:
            cued_node = memory.cue(node.vector)
            memory.store(node)
            if not cued_node:
                continue
            for csm_node in self.nodes:
                if get_similarity(cued_node.vector, csm_node.vector) == 1.0:
                    csm_node.tags.extend(cued_node.tags)

    def get_all_nodes(self):
        return list(self.nodes)

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

    def focus_vector_coalition(self, csm):
        most_similar_node, similiarity = get_most_similar_node(self.focus_vector, csm.get_all_nodes())
        return most_similar_node, similiarity 

    def focus_text_coalition(self, csm):
        pass

    def focus_tag_coalition(self, csm):
        nodes = []
        for node in csm.get_all_nodes():
            if self.focus_tag in node.tags:
                nodes.append(node)
        return max(nodes, key=lambda node: node.activation), 1.0

    def form_coalition(self, csm):
        coalition = Coalition([], self.activation)
        focus_node, similiarity = self.focus(csm)
        coalition.add_node(focus_node) 
        coalition.activation *= similiarity
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

    def decay(self):
        for coalition in self.coalitions:
            coalition.activation *= 0.9

    def run(self, incoming_coalition):
        self.decay()
        self.receive_coalition(incoming_coalition)
        winning_coalition = self.competition()
        return winning_coalition