# LIDA 

## Usage

### Sensors
Define sensors in the following format:

```python
sensors = [
    {"name": "vision_sensor1", "modality": "image"},
    {"name": "vision_sensor2", "modality": "image"},
    {"name": "hand_location", "modality": "internal_state"}
]
```

For more details on available sensors and their default configurations, please refer to the [lidapy/sensors.py](./sensors.py) file.

### Feature Detectors
Define feature detectors as either a dictionary of functions with their corresponding names or a function or list of functions (dictionary with name as key will be automatically created, useful for debugging):

```python
def face_detector(sensory_stimuli):
    '''
    sensory_stimuli: dict
    {
        "vision_sensor1": image_data,
        "vision_sensor2": image_data,
        "hand_location": internal_state_data
    }
    '''
    # Your face detection logic here
    # TODO return content instead of Node, node creation logic should be abstracted
    return Node(content=detected_face)

def object_detector(sensory_stimuli):
    ''' similar to face_detector '''
    pass

feature_detectors = {
    "face_detector": face_detector,
    "object_detector": object_detector
}
```

### Actuators
Actuators are like motors that can be controlled by the system. They can be used to perform actions based on the commands emitted by motor_plans.

Define actuators in the following format:

```python
actuators = [
    {"name": "speech_actuator", "modality": "speech"},
    {"name": "motor_actuator", "modality": "motor"}
]
```

For more details on actuators, please refer to the [lidapy/actuators.py](./actuators.py) file.

### Motor Plans
Motor plans emit sequences of motor commands based on the current state of the sensor data as defined by the dorsal stream input.

Define motor plans in the following format:

```python
actuators = [{'name': 'move'}]
actions = ['left', 'right', 'up', 'down']
def random_move(dorsal_update):
    # Your random move logic here
    return {'move': random.choice(range(len(actions)))}

motor_plans = [MotorPlan('random_move', random_move)]
```