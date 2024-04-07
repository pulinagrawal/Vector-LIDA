class Action:
    def __init__(self, params=None):
        self.params = params
    
    def execute(self):
        pass

class MotorPlanExecution:
    def __init__(self):
        pass

    def execute(self, action):
        action.execute()