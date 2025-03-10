import types
import random
from lidapy.ps import SchemeUnit
from lidapy.actuators import MotorPlan
from lidapy.utils import get_logger

logger = get_logger(__name__)

class MotorPlanExecution:
    ''' Enable to run multiple motor plans simultaneously'''
    def __init__(self, actuators):
        self.logger = get_logger(self.__class__.__name__)

        self.dorsal_update = None
        self.actuators = actuators
        self.current_plans = []
        self.selected_plan = None
        self.logger.debug("Initialized motor plan execution")

    def run(self, selected_motor_plan, dorsal_update):
        self.selected_plan = selected_motor_plan
        self.dorsal_update = dorsal_update
        if self.current_plans is None:
            self.logger.warning("No current plans available")
            return None
        if not selected_motor_plan in self.current_plans:
            self.current_plans.append(selected_motor_plan)
            self.logger.debug(f"Added new motor plan: {selected_motor_plan.name}")

        for plan in self.current_plans:
            try:
                plan.emit_command(self.dorsal_update)
            except StopIteration:
                self.current_plans.remove(plan)
                self.logger.debug(f"Removed completed plan: {plan.name}")
        
        subsumption_winner = self._run_subsumption()
        if subsumption_winner is None:
            self.logger.debug(f"No subsumption winner, using selected plan: {self.selected_plan.name}")
            return self.selected_plan.get_current_command()

        self.logger.debug(f"Subsumption winner: {subsumption_winner.name}")
        current_commands = subsumption_winner.get_current_command()
        # TODO: at some point we should make it such that results from different motor plans can be merged
        return current_commands
    
    def _run_subsumption(self):
        pass

    def _get_current_commands(self):
        return [plan.get_current_command() for plan in self.current_plans]


class SensoryMotorMemory:
    def __init__(self, motor_plans):
        self.logger = get_logger(self.__class__.__name__)
        self.motor_plans = motor_plans
        self.logger.debug(f"Initialized with {len(motor_plans)} motor plans")
      
    def cue(self, selected_motor_plan, dorsal_update):
        """
        Cues the sensory motor memory with a selected behavior and dorsal update.
        
        Args:
            selected_motor_plan: The motor plan selected by the procedural system
            dorsal_update: The current sensory input nodes
            
        Returns:
            The motor plan modified by the dorsal update
        """
        if selected_motor_plan:
            self.logger.debug(f"Cueing with motor plan: {selected_motor_plan.name}")
        return selected_motor_plan

class SensoryMotorSystem:
    def __init__(self, actuators, sensory_motor_memory, motor_plan_execution=None):
        self.logger = get_logger(self.__class__.__name__)
        if motor_plan_execution is None:
            motor_plan_execution = MotorPlanExecution(actuators)

        self.motor_plan_execution = motor_plan_execution
        self.sensory_motor_memory = sensory_motor_memory
        self.actuators = actuators
        self.logger.debug("Initialized sensory motor system")

    def run(self, selected_motor_plan=None, dorsal_update=None):
        # The dorsal stream update can trigger a change in the selected motor command in the motor plan
        self.current_motor_plan = self.sensory_motor_memory.cue(selected_motor_plan, dorsal_update)

        # select a random motor plan if no plan is selected
        if self.current_motor_plan is None:
            self.current_motor_plan = random.choice(self.sensory_motor_memory.motor_plans)
            self.logger.debug(f"No plan selected, randomly chose: {self.current_motor_plan.name}")

        self.current_motor_commands = self.motor_plan_execution.run(self.current_motor_plan, dorsal_update)
        if any(actuator not in self.actuators for actuator in self.current_motor_commands):
            self.logger.warning(f"Actuator {actuator} not in {self.actuators}")

        self.logger.info(f"Executing motor commands: {self.current_motor_commands}")
    
    def get_motor_commands(self):
        return self.current_motor_commands


