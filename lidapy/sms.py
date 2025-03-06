import types
import random
from lidapy.ps import SchemeUnit
from lidapy.actuators import MotorPlan

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


class SensoryMotorMemory:
    def __init__(self, motor_plans):
        self.motor_plans = motor_plans
      
    def cue(self, selected_motor_plan, dorsal_update):
        """
        Cues the sensory motor memory with a selected behavior and dorsal update.
        
        Args:
            selected_motor_plan: The motor plan selected by the procedural system
            dorsal_update: The current sensory input nodes
            
        Returns:
            The motor plan modified by the dorsal update
        """
        return selected_motor_plan

class SensoryMotorSystem:
    def __init__(self, sensory_motor_memory, motor_plan_execution=MotorPlanExecution()):
        self.motor_plan_execution = motor_plan_execution
        self.sensory_motor_memory = sensory_motor_memory

    def run(self, selected_motor_plan=None, dorsal_update=None):
        # The dorsal stream update can trigger a change in the selected motor command in the motor plan
        self.current_motor_plan = self.sensory_motor_memory.cue(selected_motor_plan, dorsal_update)

        # select a random motor plan if no plan is selected
        if self.current_motor_plan is None:
            self.current_motor_plan = random.choice(self.sensory_motor_memory.motor_plans)

        self.motor_plan_execution.select_plan(self.current_motor_plan)  
        self.current_motor_commands = self.motor_plan_execution.run()
    
    def get_motor_commands(self):
        return self.motor_plan_execution.get_current_commands()

    def dorsal_stream_update(self, activated_nodes):
        self.motor_plan_execution._dorsal_stream_update(activated_nodes)

