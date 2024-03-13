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
from torch import embedding
from sensory_memory import SensoryMemory
from helpers import EmbeddingModel, Node
from pam import PerceptualAssociativeMemory, VectorStore
from csm import CurrentSituationalModel, GlobalWorkspace, AttentionCodelet
from procedural_memory import ProceduralMemory, Schema

import numpy as np

# Initialize the embedding model
embedding_model = EmbeddingModel()
sensory_memory = SensoryMemory(embedding_model)
vector_store = VectorStore()
pam = PerceptualAssociativeMemory()
csm = CurrentSituationalModel(max_size=10, embedding_model=embedding_model)
procedural_memory = ProceduralMemory()
global_workspace = GlobalWorkspace()

class ACTIONS:
    TURN_ON = 1
    TURN_OFF = 2

context = Node(embedding_model.encode('lights'), 'bedroom', 1.0)
result1 = Node(embedding_model.encode('lights on'), 'lights on', 1.0)
result2 = Node(embedding_model.encode('lights off'), 'lights off', 1.0)
sample_schema1 = Schema(context, ACTIONS.TURN_ON, result1)  # This is a very simplistic schema
sample_schema2 = Schema(context, ACTIONS.TURN_OFF, result2)  # This is a very simplistic schema
procedural_memory.add_schema(sample_schema1)
procedural_memory.add_schema(sample_schema2)

focus_vector = embedding_model.encode("lights")

# Step 3: Attention Codelet forms a coalition
attention_codelet = AttentionCodelet(focus_vector)

# Assume we have a text input from the environment
while True:
    input_text = input("Enter a text: ")
    if input_text == "q":
        break

    # Step 1: Sensory Memory processes the input text
    node_from_text = sensory_memory.process_text(input_text)

    pam.process_node(node_from_text)
    # Step 2: CSM stores the new node
    csm.add_node(node_from_text)
    # Assume we have a focus vector for attention codelet
    coalition = attention_codelet.form_coalition(csm)
    print('Coalition: ', coalition)
    # Step 4: Coalition is sent to Global Workspace
    global_workspace.receive_coalition(coalition)
    # Step 5: Competition occurs in Global Workspace
    winning_coalition = global_workspace.competition()
    print('Winning Coalition: ', winning_coalition)

    # Assume we have some predefined schemas
    # Adding a simple schema for demonstration
    # Step 6: Procedural Memory selects an action based on the winning coalition
    selected_action = procedural_memory.select_action(winning_coalition)
    # The selected action node now contains the action to be executed.
    # You would have some mechanism to execute or further process this action as per your application's requirements.

    # Print the selected action text for demonstration
    print(f'Selected Action: {selected_action}')
