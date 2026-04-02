from .device import Device
from .functions import NullFunction


class ADevice(Device):
  ''' Device that takes an arbitrary cost function and constraints. '''
  _f = NullFunction()
  _constraints = []

  def slice(self, history):
    raise NotImplementedError(
      'ADevice does not support slice(): the arbitrary cost function and constraints '
      'cannot be automatically conditioned on history. Subclass ADevice and implement '
      'slice() manually if you need incremental re-optimization.'
    )

  def cost(self, s, p=0):
    s = s.reshape(len(self))
    return self.f(s) + (s*p).sum()

  def deriv(self, s, p=0):
    s = s.reshape(len(self))
    return self.f.deriv(s) + p

  def hess(self, s, p=0):
    s = s.reshape(len(self))
    return self.f.hess(s)

  @property
  def constraints(self):
    return Device.constraints.fget(self) + self._constraints

  @constraints.setter
  def constraints(self, constraints):
    self._constraints = constraints.copy()

  @property
  def f(self):
    return self._f

  @f.setter
  def f(self, f):
    if not hasattr(f, '__call__') and hasattr(f, 'deriv') and hasattr(f, 'hess'):
      raise ValueError('Invalid parameter type for \'f\' [%s]' % (type(f)))
    self._f = f
