import randomname
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name):
    return logging.getLogger(f"lidapy.{name}")

logger = get_logger(__name__)

random_name = lambda: randomname.get_name()

class Decayable:
    def __init__(self, items, threshold=0.01):
        self._decayable_items = items
        self.threshold = threshold
        self.logger = get_logger(self.__class__.__name__)

    def decay(self):
        initial_count = len(self._decayable_items)
        # Create a copy that works for both lists and sets
        items_copy = list(self._decayable_items)
        for item in items_copy:
            if hasattr(item, 'activation'):
                old_activation = item.activation
                item.activation *= 0.9
                if item.activation < self.threshold:
                    self._decayable_items.remove(item)
                    self.logger.debug(f"Removed item {item} with activation {item.activation:.3f}")
                else:
                    self.logger.debug(f"Decayed item {item} from {old_activation:.3f} to {item.activation:.3f}")
            else:
                raise ValueError(f"Item {item} does not have an activation attribute")
        removed_count = initial_count - len(self._decayable_items)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} items below threshold {self.threshold}")

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
    logger.debug(f"Linked nodes: {node1} <-> {node2}")
