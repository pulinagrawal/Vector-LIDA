from abc import ABC, abstractmethod

from regex import D

DEFAULT_CODELET_ACTIVATION = 1.0

class Codelet(ABC):
  def __init__(self):
      self.activation = DEFAULT_CODELET_ACTIVATION

  @abstractmethod
  def run(self, csm):
      pass