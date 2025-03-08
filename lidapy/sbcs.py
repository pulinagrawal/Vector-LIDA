from lidapy.utils import Node
from lidapy.codelet import Codelet
from lidapy.utils import get_logger

logger = get_logger(__name__)

def get_most_active_nodes(csm, activation_threshold=0.95):
  ''' Get highly active nodes from the CSM based on a threshold. '''
  nodes = csm.get_all_nodes()
  active_nodes = list(filter(lambda node: node.activation>activation_threshold, nodes))
  logger.debug(f"Found {len(active_nodes)} nodes above threshold {activation_threshold}")
  return active_nodes

def combine_nodes(nodes):
  if len(nodes) == 0:
    logger.debug("No nodes to combine")
    return []
  if len(nodes) == 1:
    logger.debug("Single node, no combination needed")
    return [nodes[0]]
  logger.debug(f"Combining {len(nodes)} nodes")
  return nodes[0].__class__.combine_nodes(nodes, type='sbc')

DEFAULT_SBC_FOCUS = get_most_active_nodes
DEFAULT_SBC_BUILD = combine_nodes

class StructureBuildingCodelet(Codelet):
    def __init__(self, focus_function=DEFAULT_SBC_FOCUS, build_function=DEFAULT_SBC_BUILD):
      self.logger = get_logger(self.__class__.__name__)

      super().__init__()
      if not callable(focus_function):
        raise ValueError("focus_function must be a function")
      self.focus_function = focus_function
      self.build_function = build_function
      self.logger.debug(f"Initialized with focus_function={focus_function.__name__}, build_function={build_function.__name__}")

    def run(self, csm) -> List[Node]:
      self.logger.debug("Running structure building codelet")
      focus_nodes = self.focus_function(csm)
      if not focus_nodes:
        self.logger.debug("No focus nodes found")
        return None
      new_structures = self.build_function(focus_nodes)
      self.logger.info(f"Built {len(new_structures)} new structures")
      return new_structures

    def __repr__(self) -> str:
        return f"StructureBuildingCodelet(focus_function={self.focus_function.__name__}, build_function={self.build_function.__name__})"
    