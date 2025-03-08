from abc import ABC, abstractmethod
from lidapy.utils import get_logger

logger = get_logger(__name__)

class Memory(ABC):

    @abstractmethod
    def find_associated_nodes(self, node):
        logger.debug(f"Finding associated nodes for: {node}")
        pass

    @abstractmethod
    def store(self, node):
        logger.debug(f"Storing node: {node}")
        pass

    @abstractmethod
    def learn(self, nodes):
        logger.debug(f"Learning from {len(nodes)} nodes")
        pass