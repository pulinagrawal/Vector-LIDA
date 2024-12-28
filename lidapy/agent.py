'''
I want to create a LIDA based agent that can learn from the environment and can be used to solve problems.
The agent gets information from the environment using its Sensory Memory and 
then processes it using its Perceptual Associative Memory. 
Its interface with the environment is text based.
The Sensory Memory is made using a Large Language Model (LLM).
Perceptual Associative Memory (PAM) is made using a Vector Store. 

Sensory Memory
The Sensory Memory/LLM converts the input text into vectors.
These vectors are packaged into a data structure called Node.
A Node is a vector with text and activation.

PAM
The Vector Store is a collection of nodes, each representing a text.
If a new text is encountered, it is added to the Vector Store/PAM as a new node, 
only if the dissimilarity of the new text with any existing text in the Vector Store is higher than a threshold.
Otherwise, the existing stored node for all the similar vectors recieves a boost to its activation weighted by 
their similarity.

Current Situational Model (CSM)
CSM is a short term memory that stores the most recent text encountered.
It has the capability to cue PAM to retrieve the most similar nodes from the Vector Store 
based on a vector cue.
What initiates the cue?

Attention Codelet
The Attention Codelet is the mechanism that initiates the cue.##
The attention codelets look for nodes in the CSM that are similar to the vector they are looking for. 
The node that is most similar is selected and any other nodes related/connected to this node are also selected.
They are then sent to the Global Workspace as a coalition.
Attention codeletes are learnt using reinforcement learning. When something positvitely reinforces the agent,
an attention codelet can be created that looks for the vector that was positively reinforced.

Global Workspace 
The Global Workspace is the module responsible for consciousness that states that consciousness is a result of
the competition between the different coalitions in the CSM. The most highly activated coalition wins and is
broadcasted to the rest of the system.

Procedural Memory
The Procedure Memory is the module responsible for action selection.
It contains the schemas that represent what should be done when.
Schema consists of a context, an action and a result.
The conscious broadcast triggers a schema that is most similar to the contents of the conscious broadcast.
That action is then executed.
'''
from types import SimpleNamespace
import warnings
from lidapy.ss import SensorySystem
from lidapy.csm import CurrentSituationalModel
from lidapy.global_workspace import GlobalWorkspace
from lidapy.episodic import EpisodicMemory
from lidapy.ps import ProceduralSystem
from lidapy.sms import SensoryMotorSystem
from lidapy.sbcs import StructureBuildingCodelet as SBC
from lidapy.acs import AttentionCodelet
from abc import ABC, abstractmethod
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np

class Environment(ABC):

    def execute(self, motor_commands):
        if not motor_commands:
            warnings.warn("No motor commands provided to execute.")

        if motor_commands:
            if len(motor_commands):
                self.run_commands(motor_commands)

        current_stimuli = self.recieve_sensory_stimuli()
        return current_stimuli

    @abstractmethod
    def run_commands(self, motor_commands):
        pass

    @abstractmethod
    def recieve_sensory_stimuli(self):
        pass

# Initialize the LIDA agent
sensory_system = SensorySystem()
episodic = EpisodicMemory()
csm = CurrentSituationalModel(max_size=10, sbcs=[SBC()], memories=[sensory_system.pam, episodic])
procedural_system = ProceduralSystem()
sensory_motor_system = SensoryMotorSystem()
global_workspace = GlobalWorkspace(attention_codelets=[AttentionCodelet()],
                                   broadcast_receivers=[csm, 
                                                        episodic,
                                                        sensory_system.pam,
                                                        procedural_system.pm])

DEFAULT_LIDA_AGENT = {
    "sensor_system": sensory_system,
    "csm": csm,
    "global_workspace": global_workspace,
    "procedural_system": procedural_system,
    "sensory_motor_system": sensory_motor_system,
}

def run_lida(environment, lida_agent=DEFAULT_LIDA_AGENT, steps=100):

    if not isinstance(environment, Environment):
        raise ValueError("environment must be an instance of Environment class")

    lida_agent = SimpleNamespace(**lida_agent)
    motor_commands = []
    for _ in range(steps):
        current_stimuli = environment.execute(motor_commands=motor_commands)

        associated_nodes = lida_agent.sensory_system.process(current_stimuli)

        lida_agent.csm.run(associated_nodes)

        winning_coalition = lida_agent.global_workspace.run(csm)

        print('Winning Coalition: ', winning_coalition)

        # Procedural Memory selects an action based on the winning coalition
        selected_action = lida_agent.procedural_system.run(winning_coalition)
        # The selected action node now contains the action to be executed.
        # You would have some mechanism to execute or further process this action as per your application's requirements.
        motor_commands = lida_agent.sensory_motor_system.run(selected_action)
