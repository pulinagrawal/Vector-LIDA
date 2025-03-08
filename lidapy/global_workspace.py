from lidapy.utils import Decayable, get_logger

logger = get_logger(__name__)

class Coalition:
    def __init__(self, nodes, attention_codelet):
        self.logger = get_logger(self.__class__.__name__)

        self.nodes = nodes
        self.attention_codelet = attention_codelet
        self.activation = self.compute_activation()
        self.coalition_node = self.combine_nodes(nodes)
        self.logger.debug(f"Created coalition with {len(nodes)} nodes and activation {self.activation:.3f}")

    def compute_activation(self):
        # Average activation of nodes in the coalition
        total_activation = sum(node.activation for node in self.nodes)
        avg_activation = total_activation / len(self.nodes) if self.nodes else 0
        # Weighted by the activation of the attention codelet
        weighted_activation = avg_activation * self.attention_codelet.activation
        self.logger.debug(f"Computed activation: {weighted_activation:.3f} (avg: {avg_activation:.3f}, codelet: {self.attention_codelet.activation:.3f})")
        return weighted_activation

    def add_nodes(self, nodes):
        self.logger.debug(f"Adding {len(nodes)} nodes to coalition")
        self.nodes.extend(nodes)

    def combine_nodes(self, nodes):
        if len(nodes) == 0:
            self.logger.debug("No nodes to combine")
            return None
        if len(nodes) == 1:
            self.logger.debug("Single node, no combination needed")
            return nodes[0]
        self.logger.debug(f"Combining {len(nodes)} nodes")
        return nodes[0].__class__.combine_nodes(nodes, type='coalition')

    def form_coalition(self, nodes):
        self.logger.debug(f"Forming coalition with {len(nodes)} nodes")
        self.add_nodes(nodes) 
        self.activation = self.compute_activation()
        self.coalition_node = self.combine_nodes(nodes)
        self.logger.info(f"Formed coalition with activation {self.activation:.3f}")

    def get_nodes(self):
        return self.nodes   
    
    def is_empty(self):
        return len(self.nodes) == 0

    def __repr__(self) -> str:
        return f"Coalition(nodes={self.nodes}, activation={self.activation:.3f})"

'''
The first trigger is a simple threshold on activation. When
any coalition arrives with an activation over threshold, a
competition is begun, with that strongly activated coalition
becoming the winner. This trigger insures that structures
with extraordinarily high salience have a high probability
of coming to consciousness, and thus becoming the content
of a global (conscious) broadcast (see Sections ‘A quick trip
through LIDA’s cognitive cycle’ and ‘Global workspace
theory’).
The second trigger occurs when the sum of the activations
of the coalitions in the Global Workspace exceeds a
collective threshold. This trigger is useful in those situations
where a lot of activity of moderate saliency is occurring, but
nothing of exceptional saliency.
A third trigger ensues when no new coalition arrives in
the Global Workspace for a specified period of time. This
trigger would apply to a very stable situation with little
going on.
The fourth, and default, trigger happens when there has
been no conscious broadcast for a specified period of time.
Even say during meditation in humans when purposefully
nothing of any saliency is occurring, consciousness does
not cease. Rather something of relatively little saliency is
broadcast
'''

class GlobalWorkspace(Decayable):
    def __init__(self, attention_codelets=None, broadcast_receivers=None):
        self.coalitions = []
        Decayable.__init__(self, self.coalitions)
        self.logger = get_logger(self.__class__.__name__)

        self.attention_codelets = []
        if attention_codelets is not None:
            for attention_codelet in attention_codelets:
                self.attention_codelets.append(attention_codelet)
            self.logger.debug(f"Added {len(attention_codelets)} attention codelets")

        self.broadcast_receivers = []
        if broadcast_receivers is not None:
            for broadcast_receiver in broadcast_receivers:
                self.broadcast_receivers.append(broadcast_receiver)
            self.logger.debug(f"Added {len(broadcast_receivers)} broadcast receivers")
        
        self.logger.debug("Initialized Global Workspace")

    def receive_coalition(self, coalition):
        if coalition is not None:  
            self.coalitions.append(coalition)
            self.logger.debug(f"Received coalition with activation {coalition.activation:.3f}")

    def competition(self):
        if not self.coalitions:
            self.logger.debug("No coalitions to compete")
            return None
        # Selecting the coalition with the highest activation
        winning_coalition = max(self.coalitions, key=lambda c: c.activation)
        self.logger.info(f"Competition winner: {winning_coalition} with activation {winning_coalition.activation:.3f}")
        return winning_coalition

    def run(self, csm):
        self.logger.debug("Running Global Workspace cycle")
        self.decay()
        for attention_codelet in self.attention_codelets:
            self.logger.debug(f"Running attention codelet: {attention_codelet.name}")
            coalition = attention_codelet.form_coalition(csm)
            self.receive_coalition(coalition)
        
        winning_coalition = self.competition()
        if winning_coalition is None:  
            self.logger.debug("No winning coalition")
            return None
            
        self.logger.info(f"Broadcasting coalition with {len(winning_coalition.nodes)} nodes")
        for broadcast_receiver in self.broadcast_receivers:
            self.logger.debug(f"Broadcasting to {broadcast_receiver.__class__.__name__}")
            broadcast_receiver.receive_broadcast(winning_coalition)
        return winning_coalition