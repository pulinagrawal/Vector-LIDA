import types
import random

class MotorPlanExecution:
    ''' Enable to run multiple motor plans simultaneously'''
    def __init__(self):
        self.dorsal_update = None
        self.selected_plans = []

    def select_plan(self, motor_plan):
        if motor_plan is not None:
            self.selected_plans.append(motor_plan)

    def run(self):
        if self.selected_plans is None:
            return None
        for plan in self.selected_plans:
            try:
                plan.emit_command(self.dorsal_update)
            except StopIteration:
                self.selected_plans.remove(plan)

    def get_current_commands(self):
        return [plan.get_current_command() for plan in self.selected_plans]

    def _dorsal_stream_update(self, activated_nodes):
        self.dorsal_update = activated_nodes

def generator_wrapper(func):
    ''' Wraps a function in a generator to allow for the use of the send() method. '''
    def wrapper():
        dorsal_update = yield
        while True:
            dorsal_update = yield func(dorsal_update)
    return wrapper

class MotorPlan:
    '''
    A motor plan is a sequence of motor commands that can be executed by the motor system.
    It is a generator that yields motor commands and can be updated by the dorsal stream based
    on the current stimulus/agent state. The dorsal stream update is a function that takes
    the current state and changes the current motor command based on the current state and policy.
    '''
    def __init__(self, name, policy):
        self.name = name
        self.policy = generator_wrapper(policy)()
        next(self.policy)
    
    def get_current_command(self):
        return self.current_command

    def emit_command(self, dorsal_update):
        self.current_command = self.policy.send(dorsal_update)
        return self.current_command
    

class SensoryMotorMemory:
    def __init__(self, motor_plans):
        self.motor_plans = motor_plans
      
    def cue(self, selected_behavior):
        return None

class SensoryMotorSystem:
    def __init__(self, sensory_motor_memory, motor_plan_execution=MotorPlanExecution()):
        self.motor_plan_execution = motor_plan_execution
        self.sensory_motor_memory = sensory_motor_memory

    def run(self, selected_behavior=None):
        self.current_motor_plan = self.sensory_motor_memory.cue(selected_behavior)

        # select a random motor plan if no plan is selected
        if self.current_motor_plan is None:
            self.current_motor_plan = random.choice(self.sensory_motor_memory.motor_plans)

        self.motor_plan_execution.select_plan(self.current_motor_plan)  
        self.current_motor_commands = self.motor_plan_execution.run()
    
    def get_motor_commands(self):
        return self.motor_plan_execution.get_current_commands()

    def dorsal_stream_update(self, activated_nodes):
        self.motor_plan_execution._dorsal_stream_update(activated_nodes)

