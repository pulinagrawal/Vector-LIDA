from lidapy.memory import Memory

class DefaultPAMMemory(Memory):
    def __init__(self):
        self.nodes = set()

    def store(self, node):
        self.nodes.add(node)
    
    def find_associated_nodes(self, node):
        self.learn([node])
        associated_nodes = []
        associated_nodes.extend([n for n in node.links])
        return associated_nodes

    def learn(self, nodes):
        for node in nodes:
            self.store(node)
