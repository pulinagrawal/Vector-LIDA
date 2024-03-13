import numpy as np
import torch
from sklearn.preprocessing import normalize
from transformers import GPT2Tokenizer, GPT2Model

class Node:
    def __init__(self, vector, text, activation):
        self.vector = vector
        self.text = text
        self.activation = activation

def get_most_similar_node(focus_vector, nodes):
    most_similar_node = None
    max_similarity = 0
    for node in nodes:
        similarity = cosine_similarity(focus_vector.numpy(), node.vector.numpy())[0][0]
        if similarity > max_similarity:
            most_similar_node = node
            max_similarity = similarity
    return most_similar_node, max_similarity

def combine_nodes(nodes, embedding_model):
    # Concatenate text from all nodes
    combined_text = " ".join(node.text for node in nodes)
    
    # Generate a new vector using the embedding model
    combined_vector = embedding_model.encode([combined_text])[0]
    
    # Normalize the combined vector (if necessary, depending on your embedding model)
    combined_vector = normalize(combined_vector.reshape(1, -1), norm='l2').flatten()
    
    # Create a new node with the combined vector and text
    # Assuming a default activation value for the new node, this can be adjusted as needed
    combined_node = Node(combined_vector, combined_text, activation=1.0)
    
    return combined_node

# Assume EmbeddingModel is a placeholder for your actual embedding model with an encode method
class EmbeddingModel:
    def __init__(self, model_name='gpt2-medium'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)

    def encode(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        # Get the vector representation (hidden state) of the input text
        with torch.no_grad():
            outputs = self.model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1)  # Average the hidden states to get a single vector
        return vector


# Usage
embedding_model = EmbeddingModel()
nodes_to_combine = [Node(np.array([0.1, 0.2, 0.3]), "Hello, world!", 1.0),
                    Node(np.array([0.4, 0.5, 0.6]), "Goodbye, world!", 1.0)]
combined_node = combine_nodes(nodes_to_combine, embedding_model)


class Node:
    def __init__(self, vector, text, activation):
        self.vector = vector
        self.text = text
        self.activation = activation

class SensoryMemory:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def process_text(self, text):
        vector = self.embedding_model.encode(text)        
        # Create a Node with the vector, text, and an initial activation value
        node = Node(vector, text, activation=1.0)
        
        return node
import numpy as np

class schema:
    def __init__(self, context, action, result):
        self.context = context  # list of nodes
        self.action = action  # node
        self.result = result  # list of nodes

class proceduralmemory:
    def __init__(self):
        self.schemas = []

    def add_schema(self, schema):
        self.schemas.append(schema)

    @staticmethod
    def compute_similarity(node1, node2):
        return np.dot(node1.vector, node2.vector) / \
               (np.linalg.norm(node1.vector) * np.linalg.norm(node2.vector))

    def select_action(self, coalition):
        max_match_score = 0
        selected_action = none

        for schema in self.schemas:
            match_score = 0
            for node in coalition.nodes:
                # compute similarity with context, action, and result nodes
                context_similarity = max((self.compute_similarity(node, ctx_node) for ctx_node in schema.context), default=0)
                action_similarity = self.compute_similarity(node, schema.action)
                result_similarity = max((self.compute_similarity(node, res_node) for res_node in schema.result), default=0)

                # sum up similarities (weighted by your choice, here equally weighted)
                match_score += context_similarity + action_similarity + result_similarity

            # Update selected action if current schema has a higher match score
            if match_score > max_match_score:
                max_match_score = match_score
                selected_action = schema.action

        return selected_action

from collections import deque
import numpy as np

class CurrentSituationalModel:
    def __init__(self, max_size, embedding_model):
        self.nodes_queue = deque(maxlen=max_size)  # Fixed-size queue to store nodes
        self.embedding_model = embedding_model # Embedding model to combine nodes

    def add_node(self, node):
        self.nodes_queue.append(node)  # Adds a new node to the right end of the queue

    def cue_pam(self, vector, pam, threshold):
        # Assume pam is an instance of PerceptualAssociativeMemory
        similar_nodes = pam.vector_store.find_similar_nodes(vector, threshold)
        combined_node = combine_nodes(similar_nodes, self.embedding_model)
        return combined_node
        
    def get_all_nodes(self):
        return list(self.nodes_queue)

class Coalition:
    def __init__(self, nodes, attention_codelet_activation):
        self.nodes = nodes
        self.attention_codelet_activation = attention_codelet_activation
        self.activation = self.compute_activation()

    def compute_activation(self):
        # Average activation of nodes in the coalition
        total_activation = sum(node.activation for node in self.nodes)
        avg_activation = total_activation / len(self.nodes) if self.nodes else 0
        # Weighted by the activation of the attention codelet
        weighted_activation = avg_activation * self.attention_codelet_activation
        return weighted_activation

    def add_node(self, node):
        self.nodes.append(node)
        self.activation = self.compute_activation()

    def get_nodes(self):
        return self.nodes   
    
    def is_empty(self):
        return len(self.nodes) == 0

class AttentionCodelet:
    def __init__(self, focus_vector, activation=1.0):
        self.focus_vector = focus_vector
        self.activation = activation  # Activation level of the attention codelet

    def form_coalition(self, csm):
        coalition = Coalition([], self.activation)
        most_similar_node, similiarity = get_most_similar_node(self.focus_vector, csm.get_all_nodes())
        most_similar_node.activation *= similiarity
        coalition.add_node(most_similar_node)
        return coalition

class GlobalWorkspace:
    def __init__(self):
        self.coalitions = []

    def receive_coalition(self, coalition):
        self.coalitions.append(coalition)

    def competition(self):
        if not self.coalitions:
            return None
        # Selecting the coalition with the highest activation
        winning_coalition = max(self.coalitions, key=lambda c: c.activation)
        return winning_coalition


import numpy as np

class Schema:
    def __init__(self, context, action, result):
        self.context = context  # list of Nodes
        self.action = action  # Node
        self.result = result  # list of Nodes

class ProceduralMemory:
    def __init__(self):
        self.schemas = []

    def add_schema(self, schema):
        self.schemas.append(schema)

    @staticmethod
    def compute_similarity(node1, node2):
        return np.dot(node1.vector, node2.vector) / \
               (np.linalg.norm(node1.vector) * np.linalg.norm(node2.vector))

    def select_action(self, coalition):
        max_match_score = 0
        selected_action = None

        for schema in self.schemas:
            match_score = 0
            for node in coalition.nodes:
                # Compute similarity with context, action, and result nodes
                context_similarity = max((self.compute_similarity(node, ctx_node) for ctx_node in schema.context), default=0)
                action_similarity = self.compute_similarity(node, schema.action)
                result_similarity = max((self.compute_similarity(node, res_node) for res_node in schema.result), default=0)

                # Sum up similarities (weighted by your choice, here equally weighted)
                match_score += context_similarity + action_similarity + result_similarity

            # Update selected action if current schema has a higher match score
            if match_score > max_match_score:
                max_match_score = match_score
                selected_action = schema.action

        return selected_action

