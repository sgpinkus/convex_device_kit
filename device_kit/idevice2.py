import numpy as np

from .device import Device
from device_kit.functions import HLQuadraticCost


class IDevice2(Device):
  ''' The particular cost curve is described by 2 params and *also* uses on min/max consumption bounds
  setting. All params may be ndarrays of len(self), or scalars.

  Provides a cost function that is additively separable and convex under the condition that p_h >= p_l.

  For a given time slot, let r_min, r_max be the min, max consumption as specified by Device.bounds
  for the time slot. p_h, p_l is the derivative quadratic section at r_max, r_min respectively.

  The cost value is indeterminate when r_max == r_min, but returns 0 in this case. Same for deriv.

  For load devices it should be the case that both p_h, p_l are -ve (with p_l < p_h). This is currently enforced as a
  validation rule.
  '''
  _p_h = 0
  _p_l = -1
  _cost_fn = None

  def __init__(self, id, length, bounds, cbounds=None, **kwargs):
    super().__init__(id, length, bounds, cbounds=None, **kwargs)
    self._cost_fn = HLQuadraticCost(self.p_l, self.p_h, self.lbounds, self.hbounds)

  def slice(self, history):
    '''Slice per-slot p_l, p_h params in addition to bounds/cbounds.
    Params must be sliced in the data dict before the constructor is called.'''
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
    for attr in ('p_h', 'p_l'):
      if attr in data:
        v = np.asarray(data[attr])
        if v.ndim > 0 and len(v) == len(self):
          data[attr] = v[T:]
    sliced = self.__class__(**{k: v for k, v in data.items() if k in self._keys})
    sliced._cost_fn = HLQuadraticCost(sliced.p_l, sliced.p_h, sliced.lbounds, sliced.hbounds)
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
  def p_h(self):
    return self._p_h

  @property
  def p_l(self):
    return self._p_l

  @p_h.setter
  def p_h(self, v):
    p_h = self._validate_param(v)
    if not (self.p_l <= p_h).all():
      raise ValueError('param p_h must be >= p_l')
    self._p_h = p_h

  @p_l.setter
  def p_l(self, v):
    p_l = self._validate_param(v)
    if not (self.p_h >= p_l).all():
      raise ValueError('param p_l must be <= p_h')
    self._p_l = p_l

  def _validate_param(self, p):
    v = np.array(p)
    if not (v.ndim == 0 or len(v) == len(self)):
      raise ValueError('param must be scalar or same length as device (%d)' % (len(self),))
    if not (v <= 0).all():
      raise ValueError('param must be <= 0')
    return v
