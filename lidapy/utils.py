import randomname
from abc import abstractmethod, ABC

random_name = lambda: randomname.get_name()

class Decayable:
    def __init__(self, items, threshold=0.01):
        self._decayable_items = items
        self.threshold = threshold

    def decay(self):
        for item in self._decayable_items[:]:  # Create a copy to iterate safely
            if hasattr(item, 'activation'):
                item.activation *= 0.9
                if item.activation < self.threshold:
                    self._decayable_items.remove(item)
            else:
                raise ValueError(f"Item {item} does not have an activation attribute")

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

def link_nodes(node1, node2):
    node1.links.append(node2)
    node2.links.append(node1)
