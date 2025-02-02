from lidapy.codelet import Codelet

def get_most_active_nodes(csm, activation_threshold=0.95):
  ''' Get highly active nodes from the CSM based on a threshold. '''
  nodes = csm.get_all_nodes()
  return list(filter(lambda node: node.activation>activation_threshold, nodes))

def combine_nodes(nodes):
  if len(nodes) == 0:
    return []
  if len(nodes) == 1:
    return [nodes[0]]
  return nodes[0].__class__.combine_nodes(nodes, type='sbc')

DEFAULT_SBC_FOCUS = get_most_active_nodes
DEFAULT_SBC_BUILD = combine_nodes

class StructureBuildingCodelet(Codelet):
    def __init__(self, focus_function=DEFAULT_SBC_FOCUS, build_function=DEFAULT_SBC_BUILD):
      super().__init__()
      if not callable(focus_function):
        raise ValueError("focus_function must be a function")
      self.focus_function = focus_function
      self.build_function = build_function

    def run(self, csm):
      focus_nodes = self.focus_function(csm)
      if not focus_nodes:
        return None
      new_strucutres = self.build_function(focus_nodes)
      return new_strucutres
    