from lidapy.memory import Memory
from lidapy.global_workspace import Coalition
from lidapy.utils import get_logger

logger = get_logger(__name__)

class PerceptualAssociativeMemory:
    def __init__(self, memory :Memory):
        self.memory = memory
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("Initialized PerceptualAssociativeMemory")
    
    def cue(self, nodes):
        self.logger.debug(f"Cueing with {len(nodes)} nodes")
        associated_nodes = []
        for node in nodes:
            associated_nodes.extend(self.memory.find_associated_nodes(node))
        self.logger.info(f"Retrieved {len(associated_nodes)} associated nodes")
        return associated_nodes

    def receive_broadcast(self, coalition :Coalition):
        self.logger.debug(f"Receiving broadcast from coalition: {coalition}")
        self.store(coalition.coalition_node)
        map(self.memory.store, coalition.get_nodes())

    def store(self, node):
        self.logger.debug(f"Storing node: {node}")
        self.memory.store(node)

