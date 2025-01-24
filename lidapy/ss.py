import warnings
from lidapy.utils import Node, embed
from lidapy.sensors import DEFAULT_PROCESSORS, DEFAULT_SENSORS

from lidapy.pam import PerceptualAssociativeMemory

class SensoryMemory:
    def __init__(self, sensors=DEFAULT_SENSORS):
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
                warnings.warn(f"Sensor {sensor['name']} does not have a processor,
                               using default processor for {sensor['modality']}")
                processor = DEFAULT_PROCESSORS[sensor['modality']]
            else:
                processor = sensor['processor']

            self.sensor_processors[sensor['name']] = processor

    def process(self, sensory_stimuli):
        nodes = []
        for sensor, value in sensory_stimuli.items():
            processed_output = self.sensor_processors[sensor](value)
            if not isinstance(processed_output, list) or not isinstance(processed_output, Node):
                warnings.warn(f"SENS_MEM: Processed output is not a list or Node: {processed_output}")

            if isinstance(processed_output, list):
                for item in processed_output:
                    if not isinstance(item, Node):
                        raise ValueError(f"SENS_MEM: Processed output from {sensor} is not a Node: {item}.
                                           Please verify the processor function's return.")

            nodes.extend(processed_output)
        return nodes

class SensorySystem:
    def __init__(self, sensory_memory=SensoryMemory(), pam=PerceptualAssociativeMemory()):
        self.sensory_memory = sensory_memory
        self.pam = pam

    def process(self, input):
        nodes = self.sensory_memory.process(input)
        associated_nodes = self.pam.process(nodes)
        return associated_nodes