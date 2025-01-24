from abc import ABC, abstractmethod
from lidapy.utils import random_name

from regex import D

DEFAULT_CODELET_ACTIVATION = 1.0

class Codelet(ABC):
  def __init__(self, name=None):
      if name is None:
        name = random_name()
      self.name = name
      self.activation = DEFAULT_CODELET_ACTIVATION

  @abstractmethod
  def run(self, csm):
      pass