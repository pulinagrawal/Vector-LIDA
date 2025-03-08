from collections import deque
from lidapy.utils import Decayable, get_logger

logger = get_logger(__name__)

class CurrentSituationalModel(Decayable):
    def __init__(self, ccq_maxlen=50, sbcs=None, memories=None):

        self.nodes = set()
        Decayable.__init__(self, self.nodes)
        self.logger = get_logger(self.__class__.__name__)

        self.ccq = deque(maxlen=ccq_maxlen)  # Conscious Contents Queue
        self.sbcs = []
        self.memories = []
        
        if sbcs is not None:
            for sbc in sbcs:
                self.sbcs.append(sbc)
            self.logger.debug(f"Added {len(sbcs)} structure building codelets")

        if memories is not None:
            for memory in memories:
                self.memories.append(memory)
            self.logger.debug(f"Added {len(memories)} memories")
        
        self.logger.debug(f"Initialized CSM with ccq_maxlen={ccq_maxlen}")

    def run_structure_building_codelets(self):
        for sbc in self.sbcs:
            # Possible infinite loop if a sbc calls run_structure_building_codelets on passed csm
            new_structures = sbc.run(csm=self)
            if not new_structures:
                self.logger.debug(f"No new structures from {sbc.name}")
                continue
            for structure in new_structures:
                self.add_node(structure)
                self.logger.debug(f"Added new structure: {structure}")

    def add_node(self, node):
        self.logger.debug(f"Adding node: {node}")
        self.nodes.add(node)  # Adds a new node to the right end of the queue

    def run(self, nodes):
        self.logger.debug(f"Running CSM with {len(nodes)} nodes")
        self.decay()
        for node in nodes:
            self.add_node(node)
        self.run_structure_building_codelets()
        self.logger.info(f"CSM now contains {len(self.nodes)} nodes")

    def receive_broadcast(self, coalition):
        self.logger.debug(f"Received broadcast coalition: {coalition}")
        node = coalition.coalition_node
        self.ccq.appendleft(node)
        self.logger.debug(f"Added coalition node to CCQ: {node}")
        self.cue_memories(node)

    def cue_memories(self, node):
        self.logger.debug(f"Cueing memories with node: {node}")
        for memory in self.memories:
            self.logger.debug(f"Storing node in memory: {memory.__class__.__name__}")
            memory.store(node)
            cued_nodes = memory.cue([node])
            if cued_nodes:
                self.logger.debug(f"Retrieved {len(cued_nodes)} nodes from memory")
                for cued_node in cued_nodes:
                    self.add_node(cued_node)

    def get_all_nodes(self):
        nodes = list(self.nodes)
        self.logger.debug(f"Retrieved {len(nodes)} nodes from CSM")
        return nodes