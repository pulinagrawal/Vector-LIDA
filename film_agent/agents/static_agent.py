from film_agent.agents.base_agent import BaseCLIPAgent

class StaticCLIPAgent(BaseCLIPAgent):
    """
    A static CLIP agent that uses fixed reference embeddings throughout
    """
    def __init__(self, environment):
        super().__init__(environment)

    def classify(self, frame, timestamp=None):
        return super().classify(frame, timestamp)
