from lidapy.utils import Node, embed
import logging

def _process_text(text):
    vector = embed(text)        
    # Create a Node with the vector, text, and an initial activation value
    node = Node(vector, text, activation=1.0, tags=["sensory"])
    logging.warning(f"SENS_MEM: Processed text: {text} [{node}]") 
    return node

def _process_image(image):
    pass

def _process_audio(audio):
    pass

def _process_touch(touch):
    pass

def _process_internal_state(state):
    pass

DEFAULT_PROCESSORS = {"text": _process_text,
                    "image": _process_image,
                    "audio": _process_audio,
                    "touch": _process_touch,
                    "internal_state": _process_internal_state,
                    }

DEFAULT_SENSORS = [{"name": "text", "modality": "text", "processor": _process_text},
                    {"name": "image", "modality": "image", "processor": _process_image},
                    {"name": "audio", "modality": "audio", "processor": _process_audio},
                    {"name": "touch", "modality": "touch", "processor": _process_touch},
                    {"name": "internal_state", "modality": "internal_state", "processor": _process_internal_state},
                   ]