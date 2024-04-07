from helpers import Node, embed
import logging

class SensoryMemory:
    def __init__(self):
        pass
    
    def process_text(self, text):
        vector = embed(text)        
        # Create a Node with the vector, text, and an initial activation value
        node = Node(vector, text, activation=1.0, tags=["sensory"])
        logging.warning(f"SENS_MEM: Processed text: {text} [{node}]") 
        return node

