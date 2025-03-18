import types
import random
from lidapy.actuators import MotorPlan
from lidapy.ps import Behavior
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
        # Store mappings as {motor_plan: [(dorsal_update, strength), ...]}
        self.motor_plan_dorsal_map = {plan: [] for plan in motor_plans}
        # Default initial strength for new associations
        self.initial_strength = 0.5
        # Learning rate for strengthening associations
        self.learning_rate = 0.1
        # Threshold for similarity matching
        self.similarity_threshold = 0.6
      
    def cue(self, selected_motor_plan, dorsal_update):
        """
        Cues the sensory motor memory with a selected behavior and dorsal update.
        
        Args:
            selected_motor_plan: The motor plan selected by the procedural system
            dorsal_update: The current sensory input nodes (list of nodes)
            
        Returns:
            The motor plan modified by the dorsal update
        """
        if selected_motor_plan:
            self.logger.debug(f"Cueing with motor plan: {selected_motor_plan.name}")
        if dorsal_update:
            self.logger.debug(f"Cueing with dorsal update: {dorsal_update}")
        
        # If we have a selected motor plan already, use it
        if selected_motor_plan:
            return selected_motor_plan
            
        # Otherwise, try to find a motor plan based on dorsal update similarity
        if dorsal_update:
            best_plan = self._find_best_motor_plan(dorsal_update)
            if best_plan:
                self.logger.debug(f"Found motor plan via similarity: {best_plan.name}")
                return best_plan
                
        return None

    def learn(self, motor_plan, dorsal_update):
        """
        Learn or strengthen association between a motor plan and dorsal update
        
        Args:
            motor_plan: The motor plan to associate
            dorsal_update: The dorsal update (list of Node objects) to associate
        """
        if not motor_plan or not dorsal_update:
            return
            
        # Check if we already have this dorsal update
        for i, (existing_update, strength) in enumerate(self.motor_plan_dorsal_map[motor_plan]):
            # Use our list similarity function to check if these lists are similar
            if self._calculate_list_similarity(dorsal_update, existing_update) > self.similarity_threshold:
                # Strengthen existing association
                new_strength = min(1.0, strength + self.learning_rate)
                self.motor_plan_dorsal_map[motor_plan][i] = (existing_update, new_strength)
                self.logger.debug(f"Strengthened association for {motor_plan.name}: {new_strength}")
                return
                
        # Add new association
        self.motor_plan_dorsal_map[motor_plan].append((dorsal_update, self.initial_strength))
        self.logger.debug(f"Created new association for {motor_plan.name}")
    
    def _calculate_node_similarity(self, node1, node2):
        """
        Calculate similarity between two nodes
        """
        # If nodes have a similarity method, use it
        if hasattr(node1, 'similarity') and callable(getattr(node1, 'similarity')):
            return node1.similarity(node2)
        
        # Otherwise compare by ID or content if available
        if hasattr(node1, 'id') and hasattr(node2, 'id'):
            return 1.0 if node1.id == node2.id else 0.0
        
        # Last resort: direct equality comparison
        return 1.0 if node1 == node2 else 0.0
    
    def _calculate_list_similarity(self, nodes1, nodes2):
        """
        Calculate similarity between two lists of nodes
        
        Returns a similarity score 0.0-1.0
        """
        if not nodes1 or not nodes2:
            return 0.0
        
        # Calculate best matches for each node in first list
        total_similarity = 0
        for node1 in nodes1:
            best_match = max([self._calculate_node_similarity(node1, node2) for node2 in nodes2], default=0.0)
            total_similarity += best_match
        
        # Average similarity across all nodes
        return total_similarity / len(nodes1)
            
    def _find_best_motor_plan(self, dorsal_update):
        """
        Find the motor plan with the strongest association to the given dorsal update
        
        Args:
            dorsal_update: Current dorsal update (list of nodes) to match against
            
        Returns:
            The best matching motor plan or None
        """
        best_plan = None
        best_score = 0.0
        
        for plan in self.motor_plans:
            for stored_update, strength in self.motor_plan_dorsal_map[plan]:
                # Calculate similarity score using our list similarity function
                similarity = self._calculate_list_similarity(dorsal_update, stored_update)
                score = similarity * strength
                
                if score > best_score:
                    best_score = score
                    best_plan = plan
                    
        # Only return the plan if it exceeds some minimum confidence
        if best_score > 0.3:
            return best_plan
        return None

class SensoryMotorSystem:
    def __init__(self, actuators, motor_plans, sensory_motor_memory=None, motor_plan_execution=None):
        self.logger = get_logger(self.__class__.__name__)
        if motor_plan_execution is None:
            motor_plan_execution = MotorPlanExecution(actuators)
        if sensory_motor_memory is None:
            sensory_motor_memory = SensoryMotorMemory(motor_plans)

        self.motor_plan_execution = motor_plan_execution
        self.sensory_motor_memory = sensory_motor_memory
        self.actuators = actuators
        self.actuator_names = [actuator['name'] for actuator in actuators]
        self.logger.debug("Initialized sensory motor system")

    def run(self, selected_behavior :Behavior=None, dorsal_update=None, winning_coalition=None):
        selected_motor_plan = None
        if winning_coalition is not None and selected_behavior is not None:
            selected_motor_plan = selected_behavior.find_action(winning_coalition)

        # The dorsal stream update can trigger a change in the selected motor command in the motor plan
        self.current_motor_plan = self.sensory_motor_memory.cue(selected_motor_plan, dorsal_update)

        # select a random motor plan if no plan is selected
        if self.current_motor_plan is None:
            self.current_motor_plan = random.choice(self.sensory_motor_memory.motor_plans)
            self.sensory_motor_memory.learn(self.current_motor_plan, dorsal_update)
            self.logger.debug(f"No plan selected, randomly chose: {self.current_motor_plan.name}")

        self.current_motor_commands = self.motor_plan_execution.run(self.current_motor_plan, dorsal_update)

        # Check if the motor commands are valid
        for actuator in self.current_motor_commands:
            if actuator not in self.actuator_names:
                self.logger.warning(f"Actuator {actuator} not in {self.actuators}")

        self.logger.info(f"Executing motor commands: {self.current_motor_commands}")
    
    def get_motor_commands(self):
        return self.current_motor_commands


