from collections import deque
from lidapy.helpers import get_similarity

class CurrentSituationalModel:
    def __init__(self, max_size, sbcs=None, memories=None):
        self.nodes = [] 
        self.ccq = deque(maxlen=max_size)  # Conscious Contents Queue
        self.sbcs = []
        if sbcs is not None:
            for sbc in sbcs:
                self.sbcs.append(sbc)

        self.memories = []
        if memories is not None:
            for memory in memories:
                self.memories.append(memory)

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

    def run(self, nodes):
        self.decay(self.nodes)
        for node in nodes:
            self.add_node(node)
            # Departure from basic LIDA model for cueing memories. 
            # Happens during conscious broadcast
            # self.cue_memories(node)
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