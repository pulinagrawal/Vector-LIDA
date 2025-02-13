import randomname
from abc import abstractmethod 

random_name = lambda: randomname.get_name()

class Node:
    def __init__(self, content, activation, tags=None, links=None):
        self.content = content
        self.activation = activation
        self.tags = tags
        self.links = links
        if links is None:
            self.links = []
        if tags is None:
            self.tags = []

    def __repr__(self) -> str:
        return f"Node(content={str(self.content)}, activation={self.activation})"

    def similarity(self, other_node):
        return self.__class__.similarity_function(self, other_node)

    @classmethod
    @abstractmethod
    def similarity_function(cls, one_node, other_node):
        pass


def link_nodes(node1, node2):
    node1.links.append(node2)
    node2.links.append(node1)
