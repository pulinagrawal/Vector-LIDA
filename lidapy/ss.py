import warnings
from lidapy.utils import Node, get_logger
from lidapy.sensors import DEFAULT_PROCESSORS, DEFAULT_SENSORS
from lidapy.pam import PerceptualAssociativeMemory

logger = get_logger(__name__)

class SensoryMemory:
    def __init__(self, sensors=DEFAULT_SENSORS):
        self.logger = get_logger(self.__class__.__name__)
        
        self.sensor_processors = {}
        for sensor in sensors:
            if 'name' not in sensor:
                raise ValueError(f"Sensor {sensor} does not have a name")
            if 'modality' not in sensor and 'processor' not in sensor:
                raise ValueError(f"Sensor {sensor} does not have a modality or processor")
            if 'processor' in sensor and not callable(sensor['processor']):
                raise ValueError(f"Processor for sensor {sensor['name']} is not callable")

            # Select processor
            if 'processor' not in sensor:
                warnings.warn(f"Sensor {sensor['name']} does not have a processor, \
                               using default processor for {sensor['modality']}")
                processor = DEFAULT_PROCESSORS[sensor['modality']]
            else:
                processor = sensor['processor']

            self.sensor_processors[sensor['name']] = processor
        self.logger.debug(f"Initialized with {len(sensors)} sensors")

    def process(self, sensory_stimuli):
        self.logger.debug(f"Processing sensory stimuli: {sensory_stimuli}")
        nodes = []
        for sensor, value in sensory_stimuli.items():
            if sensor not in self.sensor_processors:
                self.logger.warning(f"SENS_MEM: Sensor {sensor} not found in sensor processors. \
                               Using default processor for {sensor}")
                continue
            processed_output = self.sensor_processors[sensor](value)
            if not isinstance(processed_output, list) and not isinstance(processed_output, Node):
                raise ValueError(f"SENS_MEM: Processed output is not a list or Node: {processed_output}")

            if isinstance(processed_output, list):
                for item in processed_output:
                    if not isinstance(item, Node):
                        raise ValueError(f"SENS_MEM: Processed output from {sensor} is not a Node: {item}. \
                                           Please verify the processor function's return.")

            if isinstance(processed_output, Node):
                processed_output = [processed_output]
            nodes.extend(processed_output)
        self.logger.info(f"Processed {len(nodes)} nodes from sensory stimuli")
        self.logger.debug(f"Processed nodes: {nodes}")
        return nodes


class SensorySystem:
    def __init__(self, pam :PerceptualAssociativeMemory, sensory_memory :SensoryMemory=SensoryMemory()):
        self.logger = get_logger(self.__class__.__name__)
        self.sensory_memory = sensory_memory
        self.pam = pam
        self.logger.debug("Initialized Sensory System")

    def process(self, input):
        self.logger.debug("Processing input through sensory system")
        nodes = self.sensory_memory.process(input)
        associated_nodes = self.pam.cue(nodes)
        nodes.extend(associated_nodes)
        self.logger.info(f"Processed input into {len(nodes)} nodes")
        return nodes