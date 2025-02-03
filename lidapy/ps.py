from typing import List
import numpy as np
from lidapy.utils import Node 
# from lidapy.motor_plan_execution import Action

class ActionSequence:
    pass

class Scheme:
    def __init__(self, context: List[Node]=None, action: type[ActionSequence]=None, result: List[Node]=None):
        if not context: 
            context = Node([], "", 1.0)
        
        if not result: 
            result = Node([], "", 1.0)

        self.context = context  # a Node
        self.action = action  
        self.result = result  # a Node

class ProceduralMemory:
    def __init__(self):
        self.schemas = []

    def add_schema(self, schema):
        self.schemas.append(schema)
    
    def recieve_broadcast(self, coalition):
        self.learn_schemas(coalition)
        self.instatiate_schema(coalition)

    def learn_schemas(self, coalition):
        pass

    def run(self, winning_coalition):
        return self.instatiate_schema(winning_coalition)

    def instatiate_schema(self, coalition):
        # Create a action object with the schema's action and parameters
        schema: Schema = self._select_schema(coalition)
        return schema.action(coalition=coalition)

    def _select_schema(self, coalition) -> Schema:
        max_match_score = 0
        selected_schema = None

        for schema in self.schemas:
            match_score = 0
            for node in coalition.nodes:
                # Compute similarity with context, action, and result nodes
                context_similarity = get_similarity(node.vector, schema.context.vector)
                result_similarity = get_similarity(node.vector, schema.result.vector)

                # Sum up similarities (weighted by your choice, here equally weighted)
                match_score += context_similarity + result_similarity

                # Update selected action if current schema has a higher match score
                if match_score >= max_match_score:
                    max_match_score = match_score
                    selected_schema = schema

        return selected_schema

class ActionSelection:
    def __init__(self) -> None:
        self.behaviors = []

    def select_behavior(self):
        selected_action = None

        pass

        yield selected_action

    def run(self, selected_schema):
        pass 

        selected_behavior = self.select_behavior()
        selected_action = next(selected_behavior)
        return selected_action


class ProceduralSystem:
    def __init__(self, procedural_memory=ProceduralMemory(), action_selection=ActionSelection()):
        self.pm = procedural_memory
        self.acs = action_selection

    def run(self, winning_coalition):
        selected_schema = self.pm.run(winning_coalition)
        selected_action = self.acs.run(selected_schema)
        return selected_action