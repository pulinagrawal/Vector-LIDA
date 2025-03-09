import randomname
import logging
from abc import abstractmethod, ABC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

LIDA_COMPONENTS = {
    'all': 'lidapy',
    'sensory': 'lidapy.ss.SensorySystem',
    'sensory_memory': 'lidapy.ss.SensoryMemory',
    'pam': 'lidapy.pam.PerceptualAssociativeMemory',
    'csm': 'lidapy.csm.CurrentSituationalModel',
    'global_workspace': 'lidapy.global_workspace.GlobalWorkspace',
    'coalition': 'lidapy.global_workspace.Coalition',
    'procedural': 'lidapy.ps.ProceduralSystem',
    'procedural_memory': 'lidapy.ps.ProceduralMemory',
    'motor': 'lidapy.sms.SensoryMotorSystem',
    'motor_memory': 'lidapy.sms.SensoryMotorMemory',
    'attention': 'lidapy.acs.AttentionCodelet',
    'sbc': 'lidapy.sbcs.StructureBuildingCodelet'
}

def configure_logging(components=None, level=logging.INFO):
    """Configure logging levels for LIDA components.
    
    Args:
        components (dict, optional): Dictionary mapping component names to desired log levels.
                                   If None, sets all components to specified level.
        level (int, optional): Default logging level to use if components is None.
                             Uses standard logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Example:
        configure_logging()  # Sets all components to INFO
        configure_logging(level=logging.DEBUG)  # Sets all components to DEBUG
        configure_logging({
            'sensory': logging.DEBUG,    # Detailed logs for sensory system
            'motor': logging.WARNING,    # Only warnings for motor system
            'pam': logging.INFO         # Normal logs for PAM
        })
    """
    if components is None:
        # Set all components to the same level
        logging.getLogger('lidapy').setLevel(level)
        return

    # Set specific levels for each component
    for component, comp_level in components.items():
        if component in LIDA_COMPONENTS:
            logging.getLogger(LIDA_COMPONENTS[component]).setLevel(comp_level)
            continue
        else:
            logging.getLogger(component).setLevel(comp_level)

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
