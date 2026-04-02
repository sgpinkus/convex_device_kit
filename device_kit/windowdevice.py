import numpy as np

from .adevice import ADevice
from .functions import *


class WindowDevice(ADevice):
  ''' Penalizes flows with values outside of some a relative window, according to the magnitude
  of flow outside the window. Not convex. Overrides ADevice but `constraint` and `f` are not params.
  `c` is a scaling factor, `w` is the window width. Really there is just one of a lot of different
  penalty functions that one could use for flows outside a window.
  NOTE: This is not actually convex, but it works suprisingly well with tuned params.
  '''
  _c = 1
  _w = None

  def __init__(self, id, length, bounds, w, cbounds=None, c=1):
    super().__init__(id, length, bounds, cbounds, f=WindowPenalty(w, c), w=w, c=c)

  def slice(self, history):
    '''Rebuild a WindowDevice with sliced bounds. Overrides ADevice.slice().'''
    from device_kit.utils import adjust_cbounds as _adjust_cbounds
    history = np.asarray(history).reshape(1, -1)
    T = history.shape[1]
    if T >= len(self):
      raise ValueError(f'History length {T} must be less than device length {len(self)}')
    return WindowDevice(
      id=self.id,
      length=len(self) - T,
      bounds=self.bounds[T:],
      w=self.w,
      cbounds=_adjust_cbounds(self.cbounds, history[0], T, len(self)),
      c=self.c,
    )

  @property
  def c(self):
    return self._c

  @property
  def w(self):
    return self._w

  @c.setter
  def c(self, c):
    self._c = c

  @w.setter
  def w(self, w):
    self._w = w


class WindowPenalty(Function):
  ''' @todo deriv() is not correct?
  '''

  def __init__(self, w, c=1):
    self.c = c
    self.w = w

  def __call__(self, r):
    return self.c*(WindowPenalty.weights(r, self.w)*r).sum()

  def deriv(self, x):
    return self.c*WindowPenalty.weights(x, self.w)

  def hess(self, x):
    return np.zeros((len(x), len(x)))

  @staticmethod
  def weights(r, w):
    return np.maximum(0, np.abs(np.arange(len(r)) - WindowPenalty.com(r)) - w/2)

  @staticmethod
  def com(r):
    ''' Center of Mass. Merely weighted avg of time-slots. '''
    return np.average(np.arange(len(r)), weights=r)
