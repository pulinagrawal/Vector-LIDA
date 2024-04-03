from helpers import Node
import logging

class SensoryMemory:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def process_text(self, text):
        vector = self.embedding_model.encode(text)        
        # Create a Node with the vector, text, and an initial activation value
        node = Node(vector, text, activation=1.0)
        logging.warning(f"SENS_MEM: Processed text: {text} [{node}]") 
        return node

