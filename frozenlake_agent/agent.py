import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import random
from types import SimpleNamespace

from lidapy.utils import Node
from lidapy.acs import AttentionCodelet
from lidapy.csm import CurrentSituationalModel
from lidapy.pam import PerceptualAssociativeMemory
from lidapy.global_workspace import GlobalWorkspace
from lidapy.ss import SensoryMemory, SensorySystem
from lidapy.agent import minimally_conscious_agent
from lidapy.ps import ProceduralMemory, ProceduralSystem, SchemeUnit
from lidapy.sms import MotorPlan, SensoryMotorMemory, SensoryMotorSystem

from frozenlake_agent.pam import DefaultPAMMemory
from frozen_lake import FrozenLakeEnvironment

# The edit distance algorithm computes the minimal number of insertions, deletions, or substitutions 
# required to transform one vision string into another. For example, for 'FHFF' and 'FHHH', an edit 
# distance of 1 indicates that the visions are very similar. This is particularly useful in the Frozen 
# Lake Environment for assessing how alike two states are, given that holes are represented by 'H' and 
# goals by 'G'.

def editDistRec(s1, s2, m, n):

    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

    # If last characters of two strings are same, nothing
    # much to do. Get the count for
    # remaining strings.
    if s1[m - 1] == s2[n - 1]:
        return editDistRec(s1, s2, m - 1, n - 1)

    # If last characters are not same, consider all three
    # operations on last character of first string,
    # recursively compute minimum cost for all three
    # operations and take minimum of three values.
    return 1 + min(editDistRec(s1, s2, m, n - 1),
                   editDistRec(s1, s2, m - 1, n),
                   editDistRec(s1, s2, m - 1, n - 1))

# Wrapper function to initiate
# the recursive calculation
def editDistance(s1, s2):
    return editDistRec(s1, s2, len(s1), len(s2))

def similarity_function(cls, one_node, other_node):
    content1 = one_node.content
    content2 = other_node.content
    edit_distance = editDistance(content1, content2)
    return 1./edit_distance

Node.similarity_function = classmethod(similarity_function) # type: ignore

vision_processor = lambda x: Node(content=x['vision_sensor'], activation=1)
reward_processor = lambda x: Node(content=x['reward'], activation=1)
sensors = [
    {"name": "vision_sensor", "modality": "image"},
    {"name": "reward", "modality": "internal_state"}
]
feature_detectors = [vision_processor]
actuators = [{"name": "move"}]
pam = PerceptualAssociativeMemory(memory=DefaultPAMMemory())
sm = SensoryMemory(sensors=sensors, feature_detectors=feature_detectors)
# single expression infinite random number iterator

actions = ['left', 'down', 'right', 'up']

def avoid_hole(dorsal_update):
    node = dorsal_update[0]
    non_hole_indices = list(filter(lambda i: node.content[i] != 'H', range(len(node.content))))  # Filter out holes
    move = random.choice(non_hole_indices)  # Choose a move from the available indices 
    return {'move': move}

def random_move(dorsal_update):
    return {'move': random.choice(range(len(actions)))}

def seek_goal(dorsal_update):
    node = dorsal_update[0]
    goal_indices = list(filter(lambda i: node.content[i] == 'G', range(len(node.content))))  # Filter out goals
    if not goal_indices:
        return None
    move = random.choice(goal_indices)  # Choose a move from the available indices
    return {'move': move}

# Next three lines allows us to define a mapping between contexts and schemes
mps = [MotorPlan('random_move', random_move),
       MotorPlan('seek_goal', seek_goal),
       MotorPlan('avoid_hole', avoid_hole)]
    
contexts = [None, 'G', 'H']
schemes = [SchemeUnit(context=[Node(content=context, activation=1)]     
                                if context else None, # type: ignore
                      action=mp) 
           for context, mp in zip(contexts, mps)] 

acs = [AttentionCodelet()]
pm = ProceduralMemory(schemes=schemes)  # type: ignore

lida_agent = {
    'sensory_system': SensorySystem(pam=pam, sensory_memory=sm),  
    'csm': CurrentSituationalModel(),
    'gw': GlobalWorkspace(attention_codelets=acs, broadcast_receivers=[pm]),
    'procedural_system': ProceduralSystem(procedural_memory=pm),
    'sensory_motor_system': SensoryMotorSystem(actuators=actuators, motor_plans=mps),
}

if __name__ == '__main__':
    env = FrozenLakeEnvironment()
    # Add logging capability to the environment
    # import logging
    # env.logger = logging.getLogger('FrozenLakeEnvironment')
    minimally_conscious_agent(env, lida_agent, steps=100)