import numpy as np

from .device import Device
from .functions import Poly2D


class GDevice(Device):
  ''' Simple generator device. Strictly produces power. Cost of power is specified by an arbitrary
  polynomial.

  GDevice takes a single parameter `cost`, which defines the coefficients of the polynomial at each
  timeslot. `cost` can be a single array of coeffs or an array of len(self) arrays of coeffs (one for
  each timeslot). Thermal generators generally have quadratic cost curves.

  Generation is always indicated by negative values. However the cost function provided should define
  *positive* costs over a *positive* range (why this?). Example [1,1,0] for cost(q) = q**2 + q

  Time variable max generation capacity is specified by setting `lbounds`. GDevice enforces that
  hbounds must be <=0 for this device (can never consume).

  Cost function for all timeslots are independent. Start up / shut down and min/max runtimes are not
  considered.

  '''
  _cost_fn = None
  _cost_d1_fn = None
  _cost_d2_fn = None
  _cost_coeffs = None

  def slice(self, history):
    '''Slice cost_coeffs (if per-slot 2D) in addition to bounds/cbounds.'''
    history = np.asarray(history).reshape(1, -1)
    T = history.shape[1]
    sliced = super().slice(history)
    coeffs = self._cost_coeffs
    if np.asarray(coeffs).ndim == 2:
      sliced.cost_coeffs = np.asarray(coeffs)[T:]
    else:
      sliced.cost_coeffs = coeffs
    return sliced

  def costv(self, s, p):
    ''' Get cost vector for s, p. '''
    return s*p + self._cost_fn(-s)

  def cost(self, s, p):
    return self.costv(s, p).sum()

  def deriv(self, s, p):
    ''' Get jacobian vector of the cost at `r`, at price `p` '''
    return p - self._cost_d1_fn(-s.reshape(len(self)))

  def hess(self, s, p=0):
    ''' Return hessian. Hessdiag == Hessian for the given cost function.
    @todo actually easy to deriv explicitly ...
    '''
    return np.diag(self._cost_d2_fn(-s.reshape(len(self))))

  @property
  def bounds(self):
    return Device.bounds.fget(self)

  @property
  def cost_coeffs(self):
    ''' Return arrays of coeffs of cost function not an actual function. '''
    return self._cost_coeffs

  @bounds.setter
  def bounds(self, bounds):
    ''' @override bounds setter to ensure hbounds <= 0. '''
    Device.bounds.fset(self, bounds)
    bounds = np.array(bounds)
    if not (self.hbounds <= 0).all():
      raise ValueError('hbounds must be <= 0')

  @cost_coeffs.setter
  def cost_coeffs(self, cost):
    self._cost_coeffs = cost
    if np.array(cost).ndim == 1:
      self._cost_fn = np.poly1d(cost)
      self._cost_d1_fn = self._cost_fn.deriv()
      self._cost_d2_fn = self._cost_fn.deriv(2)
    elif np.array(cost).ndim == 2:
      self._cost_fn = lambda x: Poly2D(cost).vector(x)
      self._cost_d1_fn = lambda x: Poly2D(cost).deriv(x)
      self._cost_d2_fn = lambda x: Poly2D(cost).hess(x)
    else:
      raise ValueError('cost param must be array with 1 or 2 dimensions.')
