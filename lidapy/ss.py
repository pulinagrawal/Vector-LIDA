from lidapy.utils import get_logger
from lidapy.sensors import DEFAULT_SENSORS, DEFAULT_FEATURE_DETECTORS
from lidapy.sensors import DEFAULT_SENSORS, DEFAULT_FEATURE_DETECTORS, feature_detector
from lidapy.pam import PerceptualAssociativeMemory

logger = get_logger(__name__)

class SensoryMemory:
    def __init__(self, sensors=DEFAULT_SENSORS, feature_detectors=DEFAULT_FEATURE_DETECTORS):
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize sensors
        self.sensors = {}
        for sensor in sensors:
            if 'name' not in sensor:
                raise ValueError(f"Sensor {sensor} does not have a name")
            if 'modality' not in sensor:
                raise ValueError(f"Sensor {sensor} does not have a modality")
            
            self.sensors[sensor['name']] = sensor
        
        # Initialize feature detectors
        if feature_detectors is None:
            self.logger.warning(f"No feature detectors provided. No sensory input will be received by the agent.")

        self.feature_detectors = {}
        if isinstance(feature_detectors, list):
            for detector in feature_detectors:
                if not callable(detector):
                    raise ValueError(f"Feature detector {detector} is not callable")
            else:
                i = 0
                for detector in feature_detectors:
                    if detector.__name__ == '<lambda>':
                        self.feature_detectors |= {f"lambda_{i}": feature_detector(detector)}
                    else:
                        self.feature_detectors |= {detector.__name__: feature_detector(detector)}
        elif isinstance(feature_detectors, dict):
            for name, detector in feature_detectors.items():
                if not callable(detector):
                    raise ValueError(f"Feature detector {detector} is not callable")
                self.feature_detectors[name] = feature_detector(detector)
        elif callable(feature_detectors):
            self.feature_detectors = {feature_detectors.__name__: feature_detector(feature_detectors)}
        else:
            raise ValueError(f"Feature detectors must be a list, dict or callable. Got {type(feature_detectors)}")

        self.logger.debug(f"Initialized with {len(sensors)} sensors and {len(self.feature_detectors)} feature detectors")

    def process(self, sensory_stimuli):
        self.logger.debug(f"Processing sensory stimuli: {sensory_stimuli}")
        nodes = []
        
        # Run all feature detectors on the sensory data
        for detector in self.feature_detectors.values():
            # Check if we have all required sensors for this detector
            result = detector(sensory_stimuli)
            
            # Process the output to ensure it's a list of Nodes
            nodes.extend(result)
        
        self.logger.info(f"Processed sensory stimuli into {len(nodes)} nodes")
        self.logger.debug(f"Processed nodes: {nodes}")
        return nodes


class SensorySystem:
    def __init__(self, pam :PerceptualAssociativeMemory=None, 
                 sensory_memory :SensoryMemory=SensoryMemory()): # type: ignore
        self.logger = get_logger(self.__class__.__name__)
        self.sensory_memory = sensory_memory
        self.pam = pam
        self.logger.debug("Initialized Sensory System")

    def process(self, input):
        self.logger.debug("Processing input through sensory system")
        nodes = self.sensory_memory.process(input)
        if self.pam is not None:
            associated_nodes = self.pam.cue(nodes)
            nodes.extend(associated_nodes)
        self.logger.info(f"Processed input into {len(nodes)} nodes")
        return nodes