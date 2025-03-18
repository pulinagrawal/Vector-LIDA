from signal import alarm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))

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

import logging

from lidapy.utils import configure_logging

configure_logging({
   'httpx': logging.WARN,
   'all': logging.WARN,
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

def process_text(text):
    return vNode(content=text, vector=embed(text), activation=1.0)

import numpy as np
vNode.similarity_function = lambda x,y: np.dot(x.vector, y.vector)

# Initialize the LIDA agent
sensors = [{'name': 'input_text', 'modality': 'text', 'processor': process_text}]
actuators = [{'name': 'console_out', 'modality': 'text', 'processor': lambda x: print(x)}]
motor_plans = [MotorPlan(name='generate_text', policy=lambda dorsal: {'console_out': generate(' '.join([x.content for x in dorsal])).response})]

from lidapy.agent import run_reactive_lida, run_alarm_lida

# reactive agent
agent = {
    "sensory_system": SensorySystem(sensory_memory=SensoryMemory(sensors=sensors)),
    "sensory_motor_system": SensoryMotorSystem(actuators=actuators, motor_plans=motor_plans),
}

# run_reactive_lida(environment=SimpleTextEnvironment(), lida_agent=agent, steps=2)

# exit(0)
# alarm agent
pam = PerceptualAssociativeMemory(memory=VectorStore())
procedural_memory = ProceduralMemory(motor_plans=motor_plans)

alarm_agent_modules = {
    "sensory_system": SensorySystem(pam, sensory_memory=SensoryMemory(sensors=sensors)),
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

run_lida(environment=SimpleTextEnvironment(), lida_agent=agent, steps=2)
