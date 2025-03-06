import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from typing import List, Union

from lidapy.utils import Node 
from lidapy.actuators import MotorPlan
from lidapy.global_workspace import Coalition

class Scheme:
    def __init__(self, 
                 context: List[Node]=None, 
                 action_stream: List[Union["Scheme", "SchemeUnit"]]=None,  # allow either
                 result: List[Node]=None):  # type: ignore
        if action_stream is None:
            raise ValueError("action_stream must be provided")
        
        # If action_stream contains a single element,
        # update context/result based on its type if not provided.
        if len(action_stream) == 1:
            single = action_stream[0]
            if context is None:
                context = single.context
            if result is None:
                result = single.result
        
        if context is None: 
            context = []
        if result is None: 
            result = []
            
        self.context = context   # List[Node]
        self.action_stream = action_stream  # List[Union[Scheme, MotorPlan]]
        self.result = result     # List[Node]
    
    def find_best_action(self, coalition: Coalition, best_score=0) -> Union[MotorPlan, float]:
        # Find the first action in the action_stream that matches the coalition
        best_action = None
        for action in self.action_stream:
            if isinstance(scheme_unit:=action, SchemeUnit):
                if (curr_score:= scheme_unit.get_context_match_score(coalition)) > best_score:
                    return scheme_unit.action, curr_score
                else:
                    continue
            elif isinstance(scheme:=action, Scheme):
                best_action, best_score = scheme.find_matching_action(coalition, best_score)
        return best_action, best_score
        
    def get_context_match_score(self, coalition) -> float:
        if len(self.context) == 0:
            return 0 # If no context, assume zero match

        match_score = 0
        for node in self.context:
            scores = [node.similarity(n) for n in coalition.get_nodes()]
            match_score += max(scores)
        match_score /= len(self.context)
        return match_score
    
    def get_match_score(self, coalition) -> float:
        return self.get_context_match_score(coalition)

    @classmethod
    def from_motor_plan(cls, motor_plan: MotorPlan, 
                        context: List[Node]=None, 
                        result: List[Node]=None):
        # Wrap motor_plan directly in the action_stream list.
        return cls(context=context, action_stream=[motor_plan], result=result)

class SchemeUnit(Scheme):
    def __init__(self, context: List[Node]=None, action: MotorPlan=None, result: List[Node]=None):  # type: ignore
        if context is None:
            context = []
        if result is None:
            result = []
        self.context = context  # a Node
        self.action = action    # a MotorPlan
        self.result = result    # a Node
        self.action_stream = [self]  # a list of MotorPlans

class ProceduralMemory:
    def __init__(self, motor_plans: List[MotorPlan]=None, schemes :List[Scheme]=None): # type: ignore
        self.schemes = []
        if motor_plans is None and schemes is None:
            raise ValueError("At least one of the motor_plans or schemes must be provided.")

        if schemes is not None:
            self.schemes += schemes

        if motor_plans is not None:
            self.schemes += [SchemeUnit(action=motor_plan) for motor_plan in motor_plans]
        
    def add_scheme(self, scheme):
        self.schemes.append(scheme)
    
    def receive_broadcast(self, coalition):
        self.learn_schemes(coalition)

    def learn_schemes(self, coalition):
        pass


    def find_best_matching_scheme(self, coalition :Node) -> Scheme:
        best_scheme = max(self.schemes, key=lambda scheme: scheme.get_context_match_score(coalition))
        return best_scheme # type: ignore

    def run(self, winning_coalition):
        self.receive_broadcast(winning_coalition)
        best_scheme = self.find_best_matching_scheme(winning_coalition)
        return best_scheme

class Behavior:
    def __init__(self, scheme: Scheme, winning_coalition: Node):
        self.scheme = Scheme(context=winning_coalition.get_nodes(), action_stream=scheme.action_stream) # type: ignore
        self.activation = scheme.get_match_score(winning_coalition)  # Initial activation based on match score
    
    def find_action(self, winning_coalition):
        best_scheme = self.scheme
        best_score = -float('inf')

        # Iterate through the action stream to find the best matching scheme
        while not isinstance(best_scheme, SchemeUnit):
            for scheme in best_scheme.action_stream:
                # If the scheme is a SchemeUnit, extract its scheme (which is a MotorPlan)
                match_score = scheme.get_context_match_score(winning_coalition)
                if match_score > best_score:
                    best_score = match_score
                    best_scheme = scheme
        
        return best_scheme.action
    

class ActionSelection:
    def __init__(self) -> None:
        self.behaviors = []

    def evaluate_behavior(self, behavior: Behavior):
        # Compute the activation level of a behavior
        # This is a stub; implement your own activation evaluation
        return behavior.activation  # Assuming behavior has an activation attribute

    def select_behavior(self):
        selected = max(self.behaviors, key=lambda b: self.evaluate_behavior(b))
        return selected    
        
    def run(self, instantiated_behavior):
        # For now, assume that instantiated_behavior is appended to behaviors.
        if instantiated_behavior is not None:
            self.behaviors.append(instantiated_behavior)
        selected_behavior = self.select_behavior()
        return selected_behavior


class ProceduralSystem:
    def __init__(self, procedural_memory, action_selection=ActionSelection()):
        self.pm = procedural_memory
        self.acs = action_selection
        self.behaviors = []

    def run(self, winning_coalition):
        best_scheme = self.pm.run(winning_coalition)
        self.behaviors.append(new_behavior:=Behavior(best_scheme, winning_coalition))
        selected_behavior = self.acs.run(new_behavior)
        motor_plan = selected_behavior.find_action(winning_coalition)
        return motor_plan
