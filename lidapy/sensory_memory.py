import warnings
from lidapy.helpers import Node, embed
import logging

class SensoryMemory:
    def __init__(self):
        self.input_processor_map = { "text": self._process_text,
                                    "image": self._process_image,
                                    "audio": self._process_audio,
                                    "touch": self._process_touch,
                                    "internal_state": self._process_internal_state,
                                   }

    def process(self, input):
        nodes = []
        for key, value in input.items():
            processed_output = self.input_processor_map[key](value)
            if not isinstance(processed_output, list) or not isinstance(processed_output, Node):
                warnings.warn(f"SENS_MEM: Processed output is not a list or Node: {processed_output}")
            nodes.extend(processed_output)
        return nodes

    def _process_text(self, text):
        vector = embed(text)        
        # Create a Node with the vector, text, and an initial activation value
        node = Node(vector, text, activation=1.0, tags=["sensory"])
        logging.warning(f"SENS_MEM: Processed text: {text} [{node}]") 
        return node

    def _process_image(self, image):
        pass

    def _process_audio(self, audio):
        pass

    def _process_touch(self, touch):
        pass

    def _process_internal_state(self, state):
        pass
