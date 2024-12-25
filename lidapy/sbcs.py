from all_code import combine_nodes
from lidapy.csm import StructureBuildingCodelet
from lidapy.helpers import create_node

def combine_nodes_for_sbc(nodes):
  ''' Combine nodes to create a new node. '''
  new_node = create_node("\n".join([node.text for node in nodes]))
  new_node.tags.extend(list(tag for node in nodes for tag in node.tags))
  new_node.activation = sum(node.activation for node in nodes) / len(nodes)
  return new_node

class ActivationSBC(StructureBuildingCodelet):
  ''' Create a class that inherits from StructureBuildingCodelet and 
  grabs few of the most active node from the CSM based on a threshold 
  and creates a new node.''' 

  def __init__(self, activation_threshold=.95, combine_nodes=combine_nodes_for_sbc):
    super().__init__()
    self.activation_threshold = activation_threshold
    self.combine_nodes = combine_nodes

  def run(self, csm):
    nodes = csm.get_all_nodes()
    highly_active_nodes = [node for node in nodes if node.activation>self.activation_threshold]
    new_structure = self.combine_nodes(highly_active_nodes)
    return new_structure
  