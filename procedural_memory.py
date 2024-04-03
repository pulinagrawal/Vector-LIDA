import numpy as np
from helpers import Node, get_similarity

class Schema:
    def __init__(self, context, action, result):
        self.context = context  # a Node
        self.action = action  # Node
        self.result = result  # a Node

class ProceduralMemory:
    def __init__(self):
        self.schemas = []

    def add_schema(self, schema):
        self.schemas.append(schema)

    @staticmethod
    def compute_similarity(node1, node2):
        return get_similarity(node1.vector, node2.vector)

    def select_action(self, coalition):
        max_match_score = 0
        selected_action = None

        for schema in self.schemas:
            match_score = 0
            for node in coalition.nodes:
                # Compute similarity with context, action, and result nodes
                context_similarity = self.compute_similarity(node, schema.context)
                result_similarity = self.compute_similarity(node, schema.result)

                # Sum up similarities (weighted by your choice, here equally weighted)
                match_score += context_similarity + result_similarity

                # Update selected action if current schema has a higher match score
                if match_score > max_match_score:
                    max_match_score = match_score
                    selected_action = schema.action

        return selected_action
