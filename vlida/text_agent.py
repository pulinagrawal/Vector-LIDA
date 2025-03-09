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
    print(f"Running commands: {motor_commands}")

  def receive_sensory_stimuli(self):
      return {'input_text': 'hehe'}

def process_text(text):
    return vNode(content=text, vector=embed(text), activation=1.0)

# Initialize the LIDA agent
sensors = [{'name': 'input_text', 'modality': 'text', 'processor': process_text}]
motor_plans = [MotorPlan(name='generate_text', policy=lambda dorsal: generate(' '.join([x.content for x in dorsal])).response)]

from lidapy.agent import run_reactive_lida, run_alarm_lida

# reactive agent
reactive_agent = {
    "sensory_system": SensorySystem(sensory_memory=SensoryMemory(sensors=sensors)),
    "sensory_motor_system": SensoryMotorSystem(sensory_motor_memory=SensoryMotorMemory(motor_plans=motor_plans)),
}

run_reactive_lida(environment=SimpleTextEnvironment(), lida_agent=reactive_agent, steps=2)

# alarm agent
pam = PerceptualAssociativeMemory(memory=VectorStore())
procedural_memory = ProceduralMemory(motor_plans=motor_plans)
procedural_system = ProceduralSystem(procedural_memory=procedural_memory)
sensory_motor_memory = SensoryMotorMemory(motor_plans=motor_plans)
sensory_motor_system = SensoryMotorSystem(sensory_motor_memory=sensory_motor_memory)

alarm_agent = {
    "sensory_system": SensorySystem(pam, sensory_memory=SensoryMemory(sensors=sensors)),
    "procedural_system": procedural_system,
    "sensory_motor_system": SensoryMotorSystem(sensory_motor_memory=SensoryMotorMemory(motor_plans=motor_plans)),
}

run_alarm_lida(environment=SimpleTextEnvironment(), lida_agent=alarm_agent, steps=2)

# full agent
sensory_system = SensorySystem(pam, sensory_memory=SensoryMemory(sensors=sensors))
procedural_memory = ProceduralMemory(motor_plans=motor_plans)
procedural_system = ProceduralSystem(procedural_memory=procedural_memory)
csm = CurrentSituationalModel(ccq_maxlen=10, sbcs=[SBC()], memories=[sensory_system.pam])
global_workspace = GlobalWorkspace(attention_codelets=[AttentionCodelet()],
                                   broadcast_receivers=[csm, 
                                                        sensory_system.pam,
                                                        procedural_system.pm])
sensory_motor_memory = SensoryMotorMemory(motor_plans=motor_plans)
sensory_motor_system = SensoryMotorSystem(sensory_motor_memory=sensory_motor_memory)

full_agent = {
    "sensory_system": sensory_system,
    "csm": csm,
    "gw": global_workspace,
    "procedural_system": procedural_system,
    "sensory_motor_system": sensory_motor_system,
}

run_lida(environment=SimpleTextEnvironment(), lida_agent=full_agent, steps=2)
