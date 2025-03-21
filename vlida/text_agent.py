from signal import alarm
import sys
from pathlib import Path
import logging


sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))


from lidapy.sensors import feature_detector
from lidapy.actuators import MotorPlan
from lidapy.pam import PerceptualAssociativeMemory
from lidapy.ss import SensorySystem, SensoryMemory
from lidapy.csm import CurrentSituationalModel
from lidapy.global_workspace import GlobalWorkspace
from lidapy.ps import ProceduralMemory, ProceduralSystem
from lidapy.sms import SensoryMotorMemory, SensoryMotorSystem
from lidapy.sbcs import StructureBuildingCodelet as SBC
from lidapy.acs import AttentionCodelet
from lidapy.agent import Environment, run_lida

from vlida.pam import VectorStore
from vlida.utils import VectorNode as vNode, embed, generate

from lidapy.utils import configure_logging

# Configure logging with more specific names for the procedural system
configure_logging({
   'httpx': logging.ERROR,  
   'all': logging.ERROR,
   # Try different possible logger names for the procedural system
   'ps': logging.DEBUG,
   # Keep other modules at WARNING
   'lidapy': logging.ERROR,
})

class SimpleTextEnvironment(Environment):
  def run_commands(self, motor_commands):
    global actuators
    print(f"Running commands: {motor_commands}")
    print("Output:")
    for command in motor_commands:
        for actuator in actuators:
            if actuator['name'] == command:
                actuator['processor'](motor_commands[command])

  def receive_sensory_stimuli(self):
      return {'input_text':input("Enter text: ")}


import numpy as np
vNode.similarity_function = lambda x,y: np.dot(x.vector, y.vector)

# Initialize the LIDA agent
@feature_detector
def text_detector(sensory_stimuli):
    # Process the input text and create a vNode
    text = sensory_stimuli['input_text']
    return vNode(content=text, vector=embed(text), activation=1.0)

sensors = [{'name': 'input_text', 'modality': 'text'}]
feature_detectors = [text_detector]
actuators = [{'name': 'console_out', 'modality': 'text', 'processor': lambda x: print(x)}]
motor_plans = [MotorPlan(name='generate_factual_text', 
                         policy=lambda dorsal: {'console_out': generate(' '.join([x.content for x in dorsal]),
                                                                        system_prompt='Be factual in your response.'
                                                                       ).response}),
               MotorPlan(name='generate_helpful_text', 
                         policy=lambda dorsal: {'console_out': generate(' '.join([x.content for x in dorsal]),
                                                                        system_prompt='Be helpful in your response.'
                                                                       ).response}),

            ]

from lidapy.agent import run_reactive_lida, run_alarm_lida

# reactive agent
agent = {
    "sensory_system": SensorySystem(sensory_memory=SensoryMemory(sensors=sensors, feature_detectors=feature_detectors)),
    "sensory_motor_system": SensoryMotorSystem(actuators=actuators, motor_plans=motor_plans),
}

# run_reactive_lida(environment=SimpleTextEnvironment(), lida_agent=agent, steps=2)

# exit(0)
# alarm agent
pam = PerceptualAssociativeMemory(memory=VectorStore())
procedural_memory = ProceduralMemory(motor_plans=motor_plans)

alarm_agent_modules = {
    "sensory_system": SensorySystem(pam, sensory_memory=SensoryMemory(sensors=sensors, feature_detectors=feature_detectors)),
    "procedural_system": ProceduralSystem(procedural_memory=procedural_memory),
}
agent |= alarm_agent_modules

# run_alarm_lida(environment=SimpleTextEnvironment(), lida_agent=agent, steps=2)

# full agent
sensory_system = agent['sensory_system']
procedural_system = agent['procedural_system']
csm = CurrentSituationalModel(ccq_maxlen=10, sbcs=[SBC()], memories=[sensory_system.pam])
global_workspace = GlobalWorkspace(attention_codelets=[AttentionCodelet()],
                                   broadcast_receivers=[csm, 
                                                        sensory_system.pam,
                                                        procedural_system.pm])

full_agent_modules = {
    "csm": csm,
    "gw": global_workspace,
}

agent |= full_agent_modules

run_lida(environment=SimpleTextEnvironment(), lida_agent=agent, steps=5)
