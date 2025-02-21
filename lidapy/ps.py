import sys
from pathlib import Path

from numpy import mat
sys.path.append(str(Path(__file__).parents[1]))

from typing import List

from lidapy.utils import Node 
from lidapy.sms import MotorPlan

class SchemeUnit:
    def __init__(self, context: List[Node]=None, action: MotorPlan=None, result: List[Node]=None): # type: ignore
        if context is None:
            context = []
        
        if result is None:
            result = []

        self.context = context  # a Node
        self.action = action  # a MotorPlan
        self.result = result  # a Node

def learn_schemes(coalition):
    # Create a new scheme with the given scheme units
    schemes
    for scheme in schemes:
        if coalition in scheme.result:
            scheme.context.append(coalition)
    return scheme

class Scheme:
    def __init__(self, context: List[Node]=None, action_stream=None, result: List[Node]=None): # type: ignore
        if len(action) == 1:
            context = action[0].context
            result = action[0].result

        if context is None: 
            context = []
        
        if result is None: 
            result = []

        self.context = context  # a Node
        self.action = action  # a list of SchemeUnits
        self.result = result  # a Node

class ProceduralMemory:
    def __init__(self, motor_plans: List[MotorPlan]=None, schemes :List[Scheme]=None, coalition_match=False): # type: ignore
        if motor_plans is not None and schemes is not None:
            raise ValueError("Both motor_plans and schemes must not be provided.")
        if motor_plans is None and schemes is None:
            self.schemes = [] 
        if schemes is not None:
            schemes = self.schemes 
        if motor_plans is not None:
            self.schemes = [Scheme(action=motor_plan) for motor_plan in motor_plans]
        
        self.instatiated_scheme = self.schemes[0] if len(self.schemes)>0 else None
        self.coalition_match = coalition_match

    def add_scheme(self, scheme):
        self.schemes.append(scheme)
    
    def receive_broadcast(self, coalition):
        self.learn_schemes(coalition)
        self.instatiated_scheme = self.instatiate_scheme(coalition)

    def learn_schemes(self, coalition):
        pass

    def instatiate_scheme(self, coalition):
        # Create a action object with the scheme's action and parameters
        scheme = self.find_best_matching_scheme(coalition)
        return scheme

    def find_best_matching_scheme(self, coalition :Node) -> Scheme:
        def get_match_score(scheme):
            match_score = 0
            for node in scheme.context:
                match_score += max(map(node.similarity, coalition.get_nodes()))
            match_score /= len(scheme.context)
            return match_score
        
        best_scheme = max(self.schemes, key=get_match_score)
        return best_scheme # type: ignore

class ActionSelection:
    def __init__(self) -> None:
        self.behaviors = []

    def select_behavior(self):
        selected_action = None

        pass

        yield selected_action

    def run(self, selected_scheme):
        pass 

        selected_behavior = self.select_behavior()
        selected_action = next(selected_behavior)
        return selected_action


class ProceduralSystem:
    def __init__(self, procedural_memory, action_selection=ActionSelection()):
        self.pm = procedural_memory
        self.acs = action_selection

    def run(self, winning_coalition):
        selected_scheme = self.pm.run(winning_coalition)
        selected_action = self.acs.run(selected_scheme)
        return selected_action

if __name__ == "__main__":
    pm = ProceduralMemory()
    ProceduralSystem(pm)