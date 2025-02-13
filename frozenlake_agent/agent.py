import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import random
import gymnasium as gym

from lidapy.pam import PerceptualAssociativeMemory
from lidapy.sms import MotorPlan, SensoryMotorMemory, SensoryMotorSystem
from lidapy.ss import SensoryMemory, SensorySystem
from lidapy.utils import Node
from lidapy.agent import Environment, run_lida
from types import SimpleNamespace

from frozenlake_agent.pam import DefaultPAMMemory

class FrozenLakeEnvironment(Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    col = 0     #Data to hold the current column the agent occupies
    row = 0     # Data to hold the current row the agent occupies
    def __init__(self, render_mode="human", size=4):
    #def __init__(self):
        #generating the frozen lake environment
        self.env = gym.make(
            'FrozenLake-v1',
            desc=None,
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
        action = motor_commands[0] if len(motor_commands) > 0 else 0
        self._step_out = self.step(action)
    
    def recieve_sensory_stimuli(self):
        if self._step_out is None:
            return {}
        state, reward, done, truncated, info, surrounding_tiles = self._step_out
        stimuli = {'vision_sensor': surrounding_tiles, 'reward': reward}
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

vision_processor = lambda x: Node(content=x, activation=1)
reward_processor = lambda x: Node(content=x, activation=1)
sensors = [
    {"name": "vision_sensor", "modality": "image", "processor": vision_processor},
    {"name": "reward", "modality": "internal_state", "processor": reward_processor}
]
pam = PerceptualAssociativeMemory(memory=DefaultPAMMemory())
sm = SensoryMemory(sensors=sensors)
# single expression infinite random number iterator
def dorsal_update(self, activated_nodes):
    content = activated_nodes[0].content
    actions = ['left', 'down', 'right', 'up']
    holes = list(map(actions.index, filter(lambda x: content[x]=='H', content)))
    if self.current_command in holes:
        self.current_command = random.choices(list(set(range(4))-set(holes)))[0]
    print(f"Activated nodes: {activated_nodes}")

mp = MotorPlan('random_move', iter(lambda: random.randint(0, 3), None), dorsal_update=dorsal_update)
smm = SensoryMotorMemory(motor_plans=[mp])

lida_agent = {
    'sensory_system': SensorySystem(pam=pam, sensory_memory=sm),  # Initialize with your sensory system
    'sensory_motor_system': SensoryMotorSystem(sensory_motor_memory=smm),
}

def run_reactive_lida(environment, lida_agent, steps=100):

    if not isinstance(environment, Environment):
        raise ValueError("environment must be an instance of Environment class")

    lida_agent = SimpleNamespace(**lida_agent)
    motor_commands = []
    current_stimuli = environment.execute([0])
    for _ in range(steps):
        current_stimuli = environment.execute(motor_commands=motor_commands)

        associated_nodes = lida_agent.sensory_system.process(current_stimuli)

        motor_commands = lida_agent.sensory_motor_system.run(selected_behavior=None)
        lida_agent.sensory_motor_system.dorsal_stream(associated_nodes)
        motor_commands = lida_agent.sensory_motor_system.get_motor_commands()


if __name__ == '__main__':
    env = FrozenLakeEnvironment()
    run_reactive_lida(env, lida_agent, steps=100)