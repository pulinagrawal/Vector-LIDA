

class MotorPlan:
    '''
    A motor plan is a sequence of motor commands that can be executed by the motor system.
    It is a generator that yields motor commands and can be updated by the dorsal stream based
    on the current stimulus/agent state. The dorsal stream update is a function that takes
    the current state and changes the current motor command based on the current state and policy.
    '''
    def __init__(self, name, policy):
        def policy_generator_wrapper(func):
            ''' Wraps a function in a generator to allow for the use of the send() method. '''
            def wrapper():
                dorsal_update = yield
                while True:
                    dorsal_update = yield func(dorsal_update)
            return wrapper

        self.name = name
        self.policy = policy_generator_wrapper(policy)()
        next(self.policy)
    
    def get_current_command(self):
        return self.current_command

    def emit_command(self, dorsal_update):
        self.current_command = self.policy.send(dorsal_update)
        return self.current_command
    