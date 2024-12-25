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
from sensory_memory import SensoryMemory
from helpers import embed, Node, create_node, generate
from pam import PerceptualAssociativeMemory, VectorStore
from csm import CurrentSituationalModel, GlobalWorkspace, AttentionCodelet, StructureBuildingCodelet
from episodic import EpisodicMemory
from procedural_memory import ProceduralMemory, Schema
from motor_plan_execution import Action
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np

class CA2StructureBuildingCodelet(StructureBuildingCodelet): 
    def run(self, csm):
        new_structures = []
        to_be_removed = []
        nodes = csm.get_all_nodes()
        for node in nodes:
            if node.text.startswith("respond to"):
                for event in nodes:
                    if 'episodic' in event.tags:
                        new_structure = create_node(node.text + "\n" + f"relate to: {event.text}")
                        new_structure.tags.extend(list(tag for tag in node.tags+event.tags))
                        new_structure.activation = (node.activation + event.activation) / 2
                        new_structures.append(new_structure)
                        to_be_removed.append(node)
                        to_be_removed.append(event)
        for node in to_be_removed:
            if node in csm.nodes:
                csm.nodes.remove(node)
        return new_structures

# Initialize the embedding model
sensory_memory = SensoryMemory()
vector_store = VectorStore()
pam = PerceptualAssociativeMemory()
episodic = EpisodicMemory()
csm = CurrentSituationalModel(max_size=10, sbcs=[CA2StructureBuildingCodelet()], memories=[pam, episodic])
procedural_memory = ProceduralMemory()
global_workspace = GlobalWorkspace()

class Environment:
    def __init__(self):
        self.text = ""
    
    def execute(self, action):
        result = None
        if action:
            result = action.execute(self)

        input_text = input("Enter a text: ")
        return f"respond to: {input_text}", result
    
    def print_text(self, text):
        print(text)

class RespondAction(Action):
    def execute(self, environment):
        result = generate(prompt=self.params["context"]+"\n"+self.params["coalition"])['response']
        environment.print_text(result)
        return result


schema = Schema(action=RespondAction)  # This is a very simplistic schema
procedural_memory.add_schema(schema)

# focus_vector = embed("needs a response")

# Step 3: Attention Codelet forms a coalition
attention_codelet = AttentionCodelet(focus_tag='environment')
env = Environment()
selected_action = None

# Assume we have a text input from the environment

def run_lida(input):
    # Step 1: Sensory Memory processes the input text
    if not input:
        return 
    nodes = sensory_memory.process(input)

    pam.process(nodes)
    # Step 2: CSM stores the new node
    csm.run(nodes)
    # Assume we have a focus vector for attention codelet
    coalition = attention_codelet.form_coalition(csm)
    print('Coalition: ', coalition)
    # Step 4: Coalition is sent to Global Workspace
    winning_coalition = global_workspace.run(coalition)
    # Step 5: Competition occurs in Global Workspace
    csm.receive_broadcast(winning_coalition)
    print('Winning Coalition: ', winning_coalition)

    # Assume we have some predefined schemas
    # Adding a simple schema for demonstration
    # Step 6: Procedural Memory selects an action based on the winning coalition
    selected_action = procedural_memory.instatiate_schema(winning_coalition)
    # The selected action node now contains the action to be executed.
    # You would have some mechanism to execute or further process this action as per your application's requirements.

    # Print the selected action text for demonstration
    print(f'Selected Action: {selected_action}')
    return selected_action

while True:

    input_text, result = env.execute(selected_action)
    if input_text == "q":
        break
    if result:
        run_lida({'text': result})
    selected_action = run_lida(input_text)
