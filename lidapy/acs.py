from cv2 import sort
from httpx import get
from lidapy.codelet import Codelet
from lidapy.global_workspace import Coalition

def get_most_active_node(csm):
  ''' Get highly active nodes from the CSM based on a threshold. '''
  nodes = csm.get_all_nodes()
  return [sorted(nodes, key=lambda node: node.activation, reverse=True)[0]], None

DEFAULT_ATTENTION_CODELET = get_most_active_node

class AttentionCodelet(Codelet):
    def __init__(self, focus_function=DEFAULT_ATTENTION_CODELET):
      super().__init__()
      if not callable(focus_function):
        raise ValueError("focus_function must be a function")
      self.focus = focus_function

    def run(self, csm):
      return self.form_coalition(csm)

    def form_coalition(self, csm):
      coalition = Coalition(nodes=[], attention_codelet=self)
      focus_nodes, activation = self.focus(csm)
      coalition.add_nodes(focus_nodes) 
      if activation:
        coalition.activation = activation
      return coalition