from lidapy.codelet import Codelet
from functools import partial
from lidapy.utils import combine_nodes

def get_most_active_nodes(csm, activation_threshold=0.95):
  ''' Get highly active nodes from the CSM based on a threshold. '''
  nodes = csm.get_all_nodes()
  return list(filter(iterable=nodes, function=lambda node: node.activation>activation_threshold))

DEFAULT_SBC_FOCUS = get_most_active_nodes
DEFAULT_SBC_BUILD = lambda nodes: combine_nodes(nodes, method='average')

class StructureBuildingCodelet(Codelet):
    def __init__(self, focus_function=DEFAULT_SBC_FOCUS, build_function=DEFAULT_SBC_BUILD):
      super().__init__()
      if not callable(focus_function):
        raise ValueError("focus_function must be a function")
      self.focus_function = focus_function
      self.build_function = build_function

    def run(self, csm):
      focus_nodes = self.focus_function(csm)
      new_strucutre = self.build_function(focus_nodes)
      return new_strucutre
    