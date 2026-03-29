import numpy as np
from device_kit import Device
from device_kit.functions import ABCCost


class IDevice(Device):
  _a = 0
  _b = 2
  _c = 1
  _cost_fn = None

  def __init__(self, id, length, bounds, cbounds=None, **kwargs):
    super().__init__(id, length, bounds, cbounds=None, **kwargs)
    self._cost_fn = ABCCost(self.a, self.b, self.c, self.lbounds, self.hbounds)

  def costv(self, s, p):
    return self._cost_fn(s)/len(self) + s*p

  def cost(self, s, p):
    return self.costv(s, p).sum()

  def deriv(self, s, p):
    return self._cost_fn.deriv(s) + p

  def hess(self, s, p=0):
    return self._cost_fn.hess(s)

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @property
  def c(self):
    return self._c

  @a.setter
  def a(self, a):
    self._a = IDevice._validate_param(a, len(self))

  @b.setter
  def b(self, b):
    IDevice._validate_param(b, len(self))
    if not (np.array(b) > 0).all():
      raise ValueError('param b must be > 0')
    self._b = b

  @c.setter
  def c(self, c):
    self._c = IDevice._validate_param(c, len(self))

  @staticmethod
  def _validate_param(p, length):
    v = np.array(p)
    if not (v.ndim == 0 or len(v) == length):
      raise ValueError('param must be scalar or same length as device (%d)' % (length,))
    if not (v >= 0).all():
      raise ValueError('param must be >= 0')
    return p
