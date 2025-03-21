from lidapy.utils import Node, get_logger

logger = get_logger(__name__)

def _process_text(text):
    logger.debug(f"Processing text: {text}")
    pass

def _process_image(image):
    logger.debug(f"Processing image")
    pass

def _process_audio(audio):
    logger.debug(f"Processing audio")
    pass

def _process_touch(touch):
    logger.debug(f"Processing touch")
    pass

def _process_internal_state(state):
    logger.debug(f"Processing internal state")
    pass

DEFAULT_PROCESSORS = {"text": _process_text,
                    "image": _process_image,
                    "audio": _process_audio,
                    "touch": _process_touch,
                    "internal_state": _process_internal_state,
                    }

DEFAULT_SENSORS = [{"name": "text", "modality": "text"},
                    {"name": "eye", "modality": "image"},
                    {"name": "ear", "modality": "audio"},
                    {"name": "touch", "modality": "touch"},
                    {"name": "eye2", "modality": "image"},
                    {"name": "internal_state", "modality": "internal_state"},
                   ]

# Feature detector function decorator with metadata
def feature_detector(func):
    """
    Decorator for creating feature detectors
    
    Args:
        name (str): Name of the feature detector
        required_sensors (list): List of sensor names this detector requires
        
    Returns:
        function: Decorated function with metadata
    """
    # Add metadata to the function
    def wrapper_func(*args, **kwargs):
        result = func(*args, **kwargs)
        if not result:
            return []
        if isinstance(result, Node):
            return [result]
        if isinstance(result, list) and all(isinstance(item, Node) for item in result):
            return result
        
        logger.error(f"Feature detector {func.__name__} returned invalid result: {result}")
        return []
    wrapper_func.__name__ = func.__name__
    return wrapper_func



# Example feature detectors
@feature_detector
def audio_visual_detector(sensory_data):
    """Example detector that processes both audio and visual data"""
    logger.debug(f"Processing audio-visual data: {sensory_data}")
    # Implementation would go here
    # Return a Node or list of Nodes

@feature_detector
def multimodal_text_detector(sensory_data):
    """Example detector that processes text from multiple sources"""
    logger.debug(f"Processing multimodal text data: {sensory_data}")
    # Implementation would go here
    # Return a Node or list of Nodes

# Default feature detectors list
DEFAULT_FEATURE_DETECTORS = [
    audio_visual_detector,
    multimodal_text_detector,
]
