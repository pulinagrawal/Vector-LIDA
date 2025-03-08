from abc import ABC, abstractmethod
from lidapy.utils import random_name, get_logger
from lidapy.global_workspace import Coalition

logger = get_logger(__name__)

DEFAULT_CODELET_ACTIVATION = 1.0

class Codelet(ABC):
    def __init__(self, name=None):
        self.logger = get_logger(self.__class__.__name__)

        if name is None:
            name = random_name()
        self.name = name
        self.activation = DEFAULT_CODELET_ACTIVATION
        self.logger.debug(f"Created codelet {name} with activation {self.activation}")

    @abstractmethod
    def run(self, csm) -> Coalition:
        pass