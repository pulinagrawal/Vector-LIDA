from abc import ABC, abstractmethod

class Memory(ABC):

    @abstractmethod
    def find_associated_nodes(self, node):
        pass

    @abstractmethod
    def store(self, node):
        pass

    @abstractmethod
    def learn(self, nodes):
        pass