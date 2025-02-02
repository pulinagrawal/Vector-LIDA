import numpy as np
import logging
from lidapy.memory import Memory
from lidapy.utils import Node

class PerceptualAssociativeMemory:
    def __init__(self, memory :Memory):
        self.memory = memory
    
    def cue(self, nodes):
        associated_nodes = []
        for node in nodes:
            associated_nodes.extend(self.memory.find_associated_nodes(node))
        return associated_nodes

    def recieve_broadcast(self, coalition):
        self.store(coalition.coalition_node)
        map(self.memory.store, coalition.get_nodes())

    def store(self, node):
        self.memory.store(node)

