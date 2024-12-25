class MotorPlanExecution:
    def __init__(self):
        pass

    def process(self, motor_plan):
        pass

class SensoryMotorMemory:
    def __init__(self):
        self.motor_plans = []
      
    def cue(self, selected_action):
        pass

class SensoryMotorSystem:
    def __init__(self, motor_plan_execution=MotorPlanExecution(), sensory_motor_memory=SensoryMotorMemory()):
        self.motor_plan_execution = motor_plan_execution
        self.sensory_motor_memory = sensory_motor_memory

    def run(self, selected_action):
        motor_plan = self.sensory_motor_memory.cue(selected_action)
        motor_commands = self.motor_plan_execution.process(motor_plan)  
        return motor_commands
