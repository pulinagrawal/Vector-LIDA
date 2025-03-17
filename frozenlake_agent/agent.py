import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import random
import gymnasium as gym
from types import SimpleNamespace

from lidapy.utils import Node
from lidapy.agent import Environment 
from lidapy.acs import AttentionCodelet
from lidapy.csm import CurrentSituationalModel
from lidapy.pam import PerceptualAssociativeMemory
from lidapy.global_workspace import GlobalWorkspace
from lidapy.ss import SensoryMemory, SensorySystem
from lidapy.ps import ProceduralMemory, ProceduralSystem, SchemeUnit
from lidapy.sms import MotorPlan, SensoryMotorMemory, SensoryMotorSystem

from frozenlake_agent.pam import DefaultPAMMemory


    
class FrozenLakeEnvironment(Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    col = 0     #Data to hold the current column the agent occupies
    row = 0     # Data to hold the current row the agent occupies
    def __init__(self, render_mode="human", size=8):
    #def __init__(self):
        #generating the frozen lake environment
        self.env = gym.make(
            'FrozenLake-v1',
            desc=None,
            map_name="4x4",
            is_slippery=False,
            render_mode=render_mode)

        self.action_space = self.env.action_space  # action_space attribute
        self._step_out = None
        self.start = True
        #self.col = 0 #Agents column position
        #self.row = 0 #Agents row position

    #Reseting the environment to start a new episode
    def reset(self):
        #interacting with the environment by using Reset()
        state, info = self.env.reset()
        self.col, self.row = 0, 0 #Assuming the agent is started at (0,0)
        surrounding_tiles = self.get_surrounding_tiles(self.row, self.col)
        return state, info, surrounding_tiles, self.col, self.row

    # perform an action in environment:
    def step(self, action):
        #perform and update
        self.render()
        state, reward, done, truncated, info = self.env.step(action)
        self.update_position(state) #updating the agents position based on the action
        surrounding_tiles = self.get_surrounding_tiles(self.row, self.col)
        return state, reward, done, truncated, info, surrounding_tiles     # action chosen by the agent
        # ^returns state, reward, done, truncated, info

    def run_commands(self, motor_commands):    
        if self.start:
            self.reset()
            self.start = False
        action = motor_commands['move'] if len(motor_commands) > 0 else 0
        self._step_out = self.step(action)
    
    def receive_sensory_stimuli(self):
        if self._step_out is None:
            return {}
        state, reward, done, truncated, info, surrounding_tiles = self._step_out
        order = ['left', 'down', 'right', 'up']
        stimuli = {'vision_sensor': ''.join([surrounding_tiles[x] for x in order]),
                   'reward': reward
                  }
        return stimuli

    # render environment's current state:
    def render(self):
        self.env.render()

    # close the environment:
    def close(self):
        self.env.close()

    def update_position(self, state):
        #updating the agents position based on the action taken
        desc = self.env.unwrapped.desc
        self.row, self.col = state // desc.shape[1], state % desc.shape[1]

    def get_surrounding_tiles(self, row, col):
        #gathering information about the tiles surrounding the agent
        desc = self.env.unwrapped.desc
        surrounding_tiles = {}
        directions = {
            "up":(max(row - 1, 0), col),
            "right":(row,min(col + 1,desc.shape[1] - 1)),
            "down":(min(row + 1, desc.shape[0] - 1), col),
            "left":(row,max(col - 1, 0)),
        }
        for direction, (r,c) in directions.items():
            surrounding_tiles[direction] = desc[r,c].decode('utf-8') #Decode byte to string
        return surrounding_tiles
    
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

Node.similarity_function = classmethod(similarity_function)

vision_processor = lambda x: Node(content=x, activation=1)
reward_processor = lambda x: Node(content=x, activation=1)
sensors = [
    {"name": "vision_sensor", "modality": "image", "processor": vision_processor},
    # {"name": "reward", "modality": "internal_state", "processor": reward_processor}
]
actuators = [{"name": "move"}]
pam = PerceptualAssociativeMemory(memory=DefaultPAMMemory())
sm = SensoryMemory(sensors=sensors)
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

mps = [MotorPlan('random_move', random_move),
       MotorPlan('seek_goal', seek_goal),
       MotorPlan('avoid_hole', avoid_hole)]
    
contexts = [None, 'G', 'H']
schemes = [SchemeUnit(context=[Node(content=context, activation=1)]     
                                if context else None, # type: ignore
                      action=mp) 
           for context, mp in zip(contexts, mps)] 

acs = [AttentionCodelet()]
pm = ProceduralMemory(schemes=schemes)  # Initialize with your motor plans
# pm = ProceduralMemory(motor_plans=mps)  # Initialize with your motor plans

lida_agent = {
    'sensory_system': SensorySystem(pam=pam, sensory_memory=sm),  # Initialize with your sensory system
    'csm': CurrentSituationalModel(),
    'gw': GlobalWorkspace(attention_codelets=acs, broadcast_receivers=[pm]),
    'procedural_system': ProceduralSystem(procedural_memory=pm),
    'sensory_motor_system': SensoryMotorSystem(actuators=actuators, motor_plans=mps),
}

def run_reactive_lida(environment, lida_agent, steps=100):

    if not isinstance(environment, Environment):
        raise ValueError("environment must be an instance of Environment class")

    lida_agent = SimpleNamespace(**lida_agent)
    current_motor_commands = {'move': 0}
    current_stimuli = environment.execute(current_motor_commands)
    for _ in range(steps):
        current_stimuli = environment.execute(motor_commands=current_motor_commands)

        associated_nodes = lida_agent.sensory_system.process(current_stimuli)
        lida_agent.sensory_motor_system.dorsal_stream_update(associated_nodes)

        current_motor_commands = lida_agent.sensory_motor_system.run(selected_behavior=None)
        current_motor_commands = lida_agent.sensory_motor_system.get_motor_commands()

def run_alarm_lida(environment, lida_agent, steps=100):

    if not isinstance(environment, Environment):
        raise ValueError("environment must be an instance of Environment class")

    lida_agent = SimpleNamespace(**lida_agent)
    current_motor_commands = {'move': 0}
    current_stimuli = environment.execute(current_motor_commands)
    from lidapy.global_workspace import Coalition
    for _ in range(steps):
        current_stimuli = environment.execute(motor_commands=current_motor_commands)

        associated_nodes = lida_agent.sensory_system.process(current_stimuli)
        # lida_agent.sensory_motor_system.dorsal_stream_update(associated_nodes)

        selected_motor_plan = lida_agent.procedural_system.run(Coalition(associated_nodes, AttentionCodelet()))

        current_motor_commands = lida_agent.sensory_motor_system.run(selected_motor_plan=selected_motor_plan,
                                                             dorsal_update=associated_nodes)
        current_motor_commands = lida_agent.sensory_motor_system.get_motor_commands()

def minimally_conscious_agent(environment, lida_agent, steps=100):
    lida_agent = SimpleNamespace(**lida_agent)
    current_motor_commands = {'move': 0}
    current_stimuli = environment.execute(current_motor_commands)
    for _ in range(steps):
        current_stimuli = environment.execute(motor_commands=current_motor_commands)
        associated_nodes = lida_agent.sensory_system.process(current_stimuli)

        lida_agent.csm.run(associated_nodes)
        winning_coalition = lida_agent.gw.run(lida_agent.csm)

        selected_behavior = lida_agent.procedural_system.run(winning_coalition)
        current_motor_commands = lida_agent.sensory_motor_system.run(selected_behavior=selected_behavior, 
                                                                     dorsal_update=associated_nodes,
                                                                     winning_coalition=winning_coalition)
        current_motor_commands = lida_agent.sensory_motor_system.get_motor_commands()


if __name__ == '__main__':
    env = FrozenLakeEnvironment()
    minimally_conscious_agent(env, lida_agent, steps=100)