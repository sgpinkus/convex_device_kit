import numpy as np

from .device import Device
from .functions import ABCCost


class IDevice(Device):
  _a = 0
  _b = 2
  _c = 1
  _cost_fn = None

  def __init__(self, id, length, bounds, cbounds=None, **kwargs):
    super().__init__(id, length, bounds, cbounds=None, **kwargs)
    self._cost_fn = ABCCost(self.a, self.b, self.c, self.lbounds, self.hbounds)

  def slice(self, history):
    '''Slice per-slot params a, b, c in addition to bounds/cbounds.
    Params must be sliced in the data dict before the constructor is called,
    because __init__ validates length against self._length which is set first.'''
    history = np.asarray(history).reshape(1, -1)
    T = history.shape[1]
    if T >= len(self):
      raise ValueError(f'History length {T} must be less than device length {len(self)}')
    from device_kit.utils import adjust_cbounds as _adjust_cbounds
    h1d = history[0]
    data = self.to_dict()
    data['length'] = len(self) - T
    data['bounds'] = self.bounds[T:]
    data['cbounds'] = _adjust_cbounds(self.cbounds, h1d, T, len(self))
    for attr in ('a', 'b', 'c'):
      if attr in data:
        v = np.asarray(data[attr])
        if v.ndim > 0 and len(v) == len(self):
          data[attr] = v[T:]
    sliced = self.__class__(**{k: v for k, v in data.items() if k in self._keys})
    sliced._cost_fn = ABCCost(sliced.a, sliced.b, sliced.c, sliced.lbounds, sliced.hbounds)
    return sliced

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
