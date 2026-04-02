import numpy as np

from .device import Device


class CDevice(Device):
  ''' Overrides Device to provide a cost function based on the sum resource consumption.
    The particular cost curve is described by 2 params. C = a(Q) + b. Since Q is summed up a,b are scalars.
    b is actually useless as it doesn't effecet optimal solution.
  '''
  _a: float = 0  # Slope
  _b: float = 0  # Offset

  def cost(self, s, p):
    return (self.a*s.sum() + self.b) + (s*p).sum()

  def deriv(self, s, p):
    return np.ones(len(self))*self.a + p

  def hess(self, s, p=0):
    return np.zeros((len(self), len(self)))

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @a.setter
  def a(self, a: float):
    if a > 0:
      raise ValueError('param a must be <= 0')
    self._a = a

  @b.setter
  def b(self, b: float):
    self._b = b
