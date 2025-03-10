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

DEFAULT_SENSORS = [{"name": "text", "modality": "text", "processor": _process_text},
                    {"name": "eye", "modality": "image", "processor": _process_image},
                    {"name": "ear", "modality": "audio", "processor": _process_audio},
                    {"name": "touch", "modality": "touch", "processor": _process_touch},
                    {"name": "eye2", "modality": "image", "processor": _process_image},
                    {"name": "internal_state", "modality": "internal_state", "processor": _process_internal_state},
                   ]