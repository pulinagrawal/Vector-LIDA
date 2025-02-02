from collections import deque

class CurrentSituationalModel:
    def __init__(self, max_size, sbcs=None, memories=None):
        self.nodes = set()
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
            # Possible infinite loop if a sbc calls run_structure_building_codelets on passed csm
            new_structures = sbc.run(csm=self)
            if not new_structures:
                continue
            for structure in new_structures:
                self.add_node(structure)

    def add_node(self, node):
        self.nodes.add(node)  # Adds a new node to the right end of the queue

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

    def recieve_broadcast(self, coalition):
        node = coalition.coalition_node
        self.ccq.appendleft(node)
        self.cue_memories(node)

    def cue_memories(self, node):
        for memory in self.memories:
            memory.store(node)
            cued_nodes = memory.cue([node])
            map(self.add_node, cued_nodes)

    def get_all_nodes(self):
        return list(self.nodes)