import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))

from lidapy.sensors import DEFAULT_SENSORS
from lidapy.pam import PerceptualAssociativeMemory
from lidapy.ss import SensorySystem, SensoryMemory
from lidapy.csm import CurrentSituationalModel
from lidapy.global_workspace import GlobalWorkspace
from lidapy.ps import ProceduralSystem
from lidapy.sms import SensoryMotorSystem
from lidapy.sbcs import StructureBuildingCodelet as SBC
from lidapy.acs import AttentionCodelet
from lidapy.agent import Environment, run_lida

from vlida.pam import VectorStore
from vlida.utils import VectorNode as vNode, embed

import logging
logging.basicConfig(level=logging.INFO)


class SimpleTextEnvironment(Environment):
  def run_commands(self, motor_commands):
    print(f"Running commands: {motor_commands}")

  def recieve_sensory_stimuli(self):
      return {'input_text': 'hehe'}

def process_text(text):
    return vNode(content=text, vector=embed(text), activation=1.0)

# Initialize the LIDA agent
sensors = [{'name': 'input_text', 'modality': 'text', 'processor': process_text}]
pam = PerceptualAssociativeMemory(memory=VectorStore())
sensory_system = SensorySystem(pam, sensory_memory=SensoryMemory(sensors=sensors))
# episodic = EpisodicMemory()
csm = CurrentSituationalModel(max_size=10, sbcs=[SBC()], memories=[sensory_system.pam])
procedural_system = ProceduralSystem()
sensory_motor_system = SensoryMotorSystem()
global_workspace = GlobalWorkspace(attention_codelets=[AttentionCodelet()],
                                   broadcast_receivers=[csm, 
                                                        sensory_system.pam,
                                                        procedural_system.pm])


if __name__ == "__main__":
    agent = {
        "sensory_system": sensory_system,
        "csm": csm,
        "global_workspace": global_workspace,
        "procedural_system": procedural_system,
        "sensory_motor_system": sensory_motor_system,
    }
    run_lida(environment=SimpleTextEnvironment(), lida_agent=agent)