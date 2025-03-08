from lidapy.codelet import Codelet
from lidapy.global_workspace import Coalition
from lidapy.utils import get_logger

logger = get_logger(__name__)

def get_most_active_node(csm):
  ''' Get highly active nodes from the CSM based on a threshold. '''
  nodes = csm.get_all_nodes()
  sorted_nodes = sorted(nodes, key=lambda node: node.activation, reverse=True)[:1]
  if sorted_nodes:
    logger.debug(f"Found most active node with activation {sorted_nodes[0].activation}")
  return sorted_nodes, None

DEFAULT_ATTENTION_CODELET = get_most_active_node

class AttentionCodelet(Codelet):
    def __init__(self, focus_function=DEFAULT_ATTENTION_CODELET):
      self.logger = get_logger(self.__class__.__name__) 

      super().__init__()
      if not callable(focus_function):
        raise ValueError("focus_function must be a function")
      self.focus_function = focus_function

      self.logger.debug(f"Initialized with focus_function={focus_function.__name__}")
    
    def __repr__(self) -> str:
      return f"AttentionCodelet(focus_function={self.focus_function.__name__})"

    def run(self, csm) -> Coalition:
      self.logger.debug(f"Running attention codelet with focus_function={self.focus_function.__name__}")
      return self.form_coalition(csm)

    def form_coalition(self, csm) -> Coalition:
      coalition = Coalition(nodes=[], attention_codelet=self)
      focus_nodes, activation = self.focus_function(csm)
      if not focus_nodes:
        self.logger.debug("No focus nodes found")
        return None
      coalition.form_coalition(focus_nodes)

      self.logger.info(f"Formed coalition with {len(focus_nodes)} nodes and activation {coalition.activation:.3f}")
      self.logger.debug(f"Focus nodes: {focus_nodes}")
      self.logger.debug(f"Coalition: {coalition} with activation {coalition.activation:.3f}")
      return coalition