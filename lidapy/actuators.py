from lidapy.utils import get_logger

logger = get_logger(__name__)

class MotorPlan:
    '''
    A motor plan is a sequence of motor commands that can be executed by the motor system.
    It is a generator that yields motor commands and can be updated by the dorsal stream based
    on the current stimulus/agent state. The dorsal stream update is a function that takes
    the current state and changes the current motor command based on the current state and policy.
    '''
    def __init__(self, name, policy):
        self.logger = get_logger(self.__class__.__name__)

        self.name = name
        self.policy = policy

        def policy_generator_wrapper(func):
            ''' Wraps a function in a generator to allow for the use of the send() method. '''
            def wrapper():
                dorsal_update = yield
                while True:
                    dorsal_update = yield func(dorsal_update)
            return wrapper
        
        self.policy_generator = policy_generator_wrapper(policy)()
        next(self.policy_generator)
        self.logger.debug(f"Created motor plan {name} with policy {policy.__name__}")

    def __repr__(self) -> str:
        return f"MotorPlan(name={self.name}, policy={self.policy_generator.__name__})"
    
    def get_current_command(self):
        return self.current_command

    def emit_command(self, dorsal_update):
        self.current_command = self.policy_generator.send(dorsal_update)
        self.logger.debug(f"Motor plan {self.name} emitted command {self.current_command}")
        return self.current_command
    