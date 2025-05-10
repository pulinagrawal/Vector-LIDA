from multiprocessing import context
import sys
from pathlib import Path

from zmq import has

sys.path.append(str(Path(__file__).parents[1]))

from typing import List, Union
from itertools import product
from lidapy.utils import Node, Decayable, get_logger
from lidapy.actuators import MotorPlan
from lidapy.global_workspace import Coalition

logger = get_logger(__name__)

class Scheme:
    def __init__(self, 
                 context: List[Node]=None, 
                 action_stream: List[Union["Scheme", "SchemeUnit"]]=None,  # allow either
                 result: List[Node]=None):  # type: ignore
        self.logger = get_logger(self.__class__.__name__)
        
        if action_stream is None:
            raise ValueError("action_stream must be provided")
        
        # If action_stream contains a single element,
        # update context/result based on its type if not provided.
        if len(action_stream) == 1:
            single = action_stream[0]
            if context is None:
                context = single.context
                self.logger.debug(f"Using context from single action: {context}")
            if result is None:
                result = single.result
                self.logger.debug(f"Using result from single action: {result}")
        
        if context is None: 
            context = []
        if result is None: 
            result = []
            
        self.context = context   # List[Node]
        self.action_stream = action_stream  # List[Union[Scheme, MotorPlan]]
        self.result = result     # List[Node]
        self.logger.debug(f"Created scheme with {len(self.context)} context nodes, {len(self.action_stream)} actions, {len(self.result)} result nodes")

    def __repr__(self) -> str:
        return f"Scheme(context={self.context}, action_stream={self.action_stream}, result={self.result})"
    
    def find_best_action(self, coalition: Coalition, best_score=0) -> Union[MotorPlan, float]:
        self.logger.debug(f"Finding best action for coalition {coalition}")
        # Find the first action in the action_stream that matches the coalition
        best_action = None
        for action in self.action_stream:
            if isinstance(scheme_unit:=action, SchemeUnit):
                if (curr_score:= scheme_unit.get_context_match_score(coalition)) > best_score:
                    self.logger.debug(f"Found better action {scheme_unit.action} with score {curr_score:.3f}")
                    return scheme_unit.action, curr_score
                else:
                    continue
            elif isinstance(scheme:=action, Scheme):
                best_action, best_score = scheme.find_matching_action(coalition, best_score)
        self.logger.debug(f"Returning best action {best_action} with score {best_score:.3f}")
        return best_action, best_score
        
    def get_context_match_score(self, coalition) -> float:
        if len(self.context) == 0:
            self.logger.debug("No context, returning default score 0.3")
            return 0.05 # If no context, assume zero match

        match_score = 0
        for node in self.context:
            scores = [node.similarity(n) for n in coalition.get_nodes()]
            match_score += max(scores)
        match_score /= len(self.context)
        self.logger.debug(f"Context match score: {match_score:.3f}")
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
        self.logger = get_logger(self.__class__.__name__)

        if context is None:
            context = []
        if result is None:
            result = []
        self.context = context  # a Node
        self.action = action    # a MotorPlan
        self.result = result    # a Node
        self.action_stream = [self]  # a list of MotorPlans
    
    def __repr__(self) -> str:
        return f"SchemeUnit(context={self.context}, action={self.action}, result={self.result})"

class ProceduralMemory:
    def __init__(self, motor_plans: List[MotorPlan]=None, schemes :List[Scheme]=None): # type: ignore
        self.schemes = []
        self.logger = get_logger(self.__class__.__name__)
        
        if motor_plans is None and schemes is None:
            raise ValueError("At least one of the motor_plans or schemes must be provided.")

        if schemes is not None:
            self.schemes += schemes
            self.logger.debug(f"Added {len(schemes)} schemes from schemes parameter")

        if motor_plans is not None:
            new_schemes = [SchemeUnit(action=motor_plan) for motor_plan in motor_plans]
            self.schemes += new_schemes
            self.logger.debug(f"Created and added {len(new_schemes)} schemes from motor plans")
        
    def add_scheme(self, scheme):
        self.schemes.append(scheme)
        self.logger.debug(f"Added new scheme: {scheme}")
    
    def receive_broadcast(self, coalition):
        self.logger.debug(f"Received broadcast coalition: {coalition}")

    def learn_schemes(self, best_matching_scheme, winning_coalition):
        self.logger.debug(f"Learning new scheme from {best_matching_scheme} and coalition {winning_coalition}")
        new_context = {node.content: node.copy() for node in best_matching_scheme.context}
        for node in winning_coalition.get_nodes():
            if node.content in new_context:
                new_context[node.content].combine_features(node)
            else:
                new_context[node.content] = node.copy()
        new_scheme = Scheme(context=list(new_context.values()), 
                            action_stream=best_matching_scheme.action_stream,
                            result=best_matching_scheme.result)
        self.schemes.append(new_scheme)

    def find_best_matching_scheme(self, coalition :Node) -> Scheme:
        best_scheme = max(self.schemes, key=lambda scheme: scheme.get_context_match_score(coalition))
        self.logger.info(f"Found best matching scheme {best_scheme} with score {best_scheme.get_context_match_score(coalition):.3f}")
        return best_scheme # type: ignore

    def run(self, winning_coalition):
        self.logger.debug(f"Running with winning coalition: {winning_coalition}")
        self.receive_broadcast(winning_coalition)
        best_scheme = self.find_best_matching_scheme(winning_coalition)
        self.learn_schemes(best_scheme, winning_coalition)
        return best_scheme

class Behavior:
    def __init__(self, scheme: Scheme, winning_coalition: Node):
        self.scheme = Scheme(context=winning_coalition.get_nodes(), action_stream=scheme.action_stream) # type: ignore
        self.activation = scheme.get_match_score(winning_coalition)  # Initial activation based on match score
    
    def __repr__(self) -> str:
        return f"Behavior(scheme={self.scheme}, activation={self.activation:.3f})"
    
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
    

class ActionSelection(Decayable):
    def __init__(self) -> None:
        self.behaviors = []
        Decayable.__init__(self, self.behaviors)
        self.logger = get_logger(self.__class__.__name__)

    def evaluate_behavior(self, behavior: Behavior):
        activation = behavior.activation
        self.logger.debug(f"Evaluating behavior {behavior} with activation {activation:.3f}")
        return activation

    def select_behavior(self):
        selected = max(self.behaviors, key=lambda b: self.evaluate_behavior(b))
        self.logger.info(f"Selected behavior {selected} with activation {selected.activation:.3f}")
        return selected    
        
    def run(self, instantiated_behavior):
        self.logger.debug("Running action selection")
        self.decay()  # Apply decay to all behaviors
        if instantiated_behavior is not None:
            self.behaviors.append(instantiated_behavior)
            self.logger.debug(f"Added new behavior {instantiated_behavior}")
        selected_behavior = self.select_behavior()
        return selected_behavior


class ProceduralSystem:
    def __init__(self, procedural_memory, action_selection=ActionSelection()):
        self.pm = procedural_memory
        self.acs = action_selection
        self.behaviors = []
        self.logger = get_logger(self.__class__.__name__)

    def run(self, winning_coalition):
        self.logger.debug(f"Processing coalition {winning_coalition}")
        best_scheme = self.pm.run(winning_coalition)
        self.logger.debug(f"Selected scheme {best_scheme}")
        self.behaviors.append(new_behavior:=Behavior(best_scheme, winning_coalition))
        selected_behavior = self.acs.run(new_behavior)
        self.logger.info(f"Selected behavior {selected_behavior}")
        return selected_behavior
