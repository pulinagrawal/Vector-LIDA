import types

class MotorPlanExecution:
    ''' Enable to run multiple motor plans simultaneously'''
    def __init__(self, dorsal_update = lambda x: None):
        self.dorsal_update = dorsal_update
        self.selected_plans = []

    def select_plan(self, motor_plan):
        if motor_plan is not None:
            self.selected_plans.append(motor_plan)

    def run(self):
        if self.selected_plans is None:
            return None
        for plan in self.selected_plans:
            try:
                plan.emit_command()
            except StopIteration:
                self.selected_plans.remove(plan)

    def get_current_commands(self):
        return [plan.get_current_command() for plan in self.selected_plans]

    def dorsal_stream_update(self, activated_nodes):
        for plan in self.selected_plans:
            plan.dorsal_update(activated_nodes)

class MotorPlan:
    def __init__(self, name, motor_commands, dorsal_update):
        self.name = name
        self.motor_commands = motor_commands
        if dorsal_update:
            self.dorsal_update = types.MethodType(dorsal_update, self)
    
    def get_current_command(self):
        return self.current_command

    def emit_command(self):
        self.current_command = next(self.motor_commands)
        return self.current_command
    
    def dorsal_update(self, activated_nodes):
        pass


class SensoryMotorMemory:
    def __init__(self, motor_plans):
        self.motor_plans = motor_plans
      
    def cue(self, selected_action):
        return self.motor_plans[0]

class SensoryMotorSystem:
    def __init__(self, sensory_motor_memory, motor_plan_execution=MotorPlanExecution()):
        self.motor_plan_execution = motor_plan_execution
        self.sensory_motor_memory = sensory_motor_memory

    def run(self, selected_action=None):
        self.current_motor_plan = self.sensory_motor_memory.cue(selected_action)
        self.motor_plan_execution.select_plan(self.current_motor_plan)  
        self.current_motor_commands = self.motor_plan_execution.run()
    
    def get_motor_commands(self):
        return self.motor_plan_execution.get_current_commands()

    def dorsal_stream(self, activated_nodes):
        return self.motor_plan_execution.dorsal_stream_update(activated_nodes)
