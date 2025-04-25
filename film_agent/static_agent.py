from base_agent import BaseCLIPAgent

class StaticCLIPAgent(BaseCLIPAgent):
    def __init__(self, environment):
        super().__init__(environment)

    def classify(self, frame, timestamp=None):
        return super().classify(frame, timestamp)
