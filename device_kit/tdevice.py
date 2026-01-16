import numpy as np
import numdifftools as nd
from device_kit import Device, IDevice
from device_kit.functions import ABCCost
from device_kit.utils import soc, base_soc, sustainment_matrix


class TDevice(Device):
  ''' Represents a heating or cooling device. Or really any device who's cost is based on a
  variable that behaves like thermodynamic cooling/heating. The cost curve is the same as IDevice, but based on
  instaneous temperature not resource consumption directly, and with two restrictions on parameters:

    IDevice.a is always 0.
    IDevice.b must be even. This is just a quick fix. May change to allow odd in future.

  Recall IDevice.c is a scaling factor for cost. Thus if one doesn't care about the temperature
  at a given time, set IDevice.c to zero. If one cares a little, set IDevice.c a little.

  The instaneous temperature is dependent on resource consumption at all times before a given time.
  Under the hood, the thermodynamics are modelled by a simple discrete exponential equation:

    T(i) = T(i-1) + A(TE(i) - T(i-1)) + B(Q(t))

  Where T internal temperature, TE external temperature, Q(t) consumption at t, A, B constant params.

  Note that the min|max acceptable temps may be infeasible given TE, and consumption bounds and
  other parameters. So we haven't implemented them as hard constraints. The min|max temp inputs are
  actually just parameters to the cost function not hard constraints. Specifically the diff
  between t_optimal and t_min or t_max (symmetrically) is `c` units of cost (@see IDevice.c).

  @todo this doesn't really need to extend IDevice. It just uses a classs method from IDevice for its
  cost curve.
  '''
  _sustainment = None            # Thermal loss factor to external environment.
  _efficiency = None             # Thermal efficiency factor. This also includes the unit conversion factor.
  _t_external = None             # External temperature vectors of `len` length.
  _t_range = None                # The +/- range min/max temp.
  _c = 1                         # Scaling factor for cost function.
  t_init = None                  # The initial (and final) temperature T0.
  t_optimal = None               # The scalar ideal temperature TC. It's assumed ideal temperature is time invariant.
  t_base = None                  # temperature without out any consumption by heat engine. Derived value.
  t_cost_base = 0             # cost of t_base pre calculated & used as offset.
  sustainment_matrix = None      # stashed for use in deriv.
  _cost_fn = None

  def __init__(self, id, length, bounds, sustainment, efficiency, t_init, t_optimal, t_range, t_external, c=1, cbounds=None, **meta):
    ''' Set params and check validity. Set derived vars t_base, sustainment_min|max based on params. '''
    super().__init__(id, length, bounds, cbounds, **meta)
    if not 0 <= sustainment <= 1:
      raise ValueError('sustainment (heat transfer coefficient) must be in [0,1]')
    if efficiency == 0:
      raise ValueError('efficiency factor must not be 0')
    if len(t_external) != len(self):
      raise ValueError('external temperature vector has wrong len (%s)' % (t_external,))
    t_range = np.ones(self.shape)*t_range
    if (t_range < 0.0).any():
      raise ValueError('acceptable temperature range (t_range) must be >= 0')
    self._sustainment = sustainment
    self._efficiency = efficiency
    self._t_init = t_init
    self._t_optimal = t_optimal
    self._t_range = t_range
    self._t_external = t_external
    self._c = IDevice._validate_param(c, len(self))
    self._cost_fn = ABCCost(0, 2, self.c, self.t_min, self.t_optimal)
    # Set some computed values.
    self.t_base = self._make_t_base(self.t_external, self.sustainment, self.t_init)
    self.t_cost_base = self.costv_t(self.t_base)
    self.sustainment_matrix = sustainment_matrix(self.sustainment, len(self))
    # t_act_min|max for convenience only. The min|max temperature change that the device itself must
    # cause - not the externals to bring be within range. Achieving t_act_min|max may not actually be feasibile.
    self.t_act_min = (self.t_optimal - self.t_range) - self.t_base
    self.t_act_max = (self.t_optimal + self.t_range) - self.t_base

  def cost(self, s, p):
    return self.costv(s, p).sum()

  def costv(self, s, p):
    ''' @override uv() to do r to t conversion. '''
    return self.costv_t(self.r2t(s)) + s*p

  def deriv(self, s, p):
    ''' @override deriv() to do r to t conversion. Chain rule to account for r2t(). '''
    dt = self.deriv_t(self.r2t(s))
    return (self.sustainment_matrix*dt.reshape(len(self),1)).sum(axis=0)*self.efficiency + p

  def hess(self, s, p=0):
    ''' Return hessian diagonal approximation. nd.Hessian takes long time. In testing so far
    Hess diag is 10x faster and works just as good if not better.
    '''
    return np.diag(nd.Hessdiag(lambda x: self.cost(x, 0))(s.reshape(len(self))))

  def costv_t(self, t):
    return self._cost_fn(t)

  def deriv_t(self, t):
    return self._cost_fn.deriv(t)

  def r2t(self, r):
    ''' Map `r` consumption vector to its effective heating or cooling effect, given heat transfer
    (t_base), thermal loss (sustainment) and efficiency of device (efficiency).
    '''
    return self.t_base + soc(r.reshape(len(self)), s=self.sustainment, e=self.efficiency)

  @property
  def params(self):
    return {
      't_external': self.t_external,
      't_init': self.t_init,
      't_optimal': self.t_optimal,
      't_range': self.t_range,
      'sustainment': self.sustainment,
      'efficiency': self.efficiency
    }
    return p

  @property
  def sustainment(self):
    return self._sustainment

  @property
  def efficiency(self):
    return self._efficiency

  @property
  def t_init(self):
    return self._t_init

  @property
  def t_optimal(self):
    return self._t_optimal

  @property
  def t_range(self):
    return self._t_range

  @property
  def t_external(self):
    return self._t_external

  @property
  def c(self):
    return self._c

  @property
  def t_min(self):
    return self.t_optimal - self.t_range

  @property
  def t_max(self):
    return self.t_optimal + self.t_range

  def _make_t_base(self, t_external, sustainment, t_init):
    ''' Calculate the base temperature, that occurs with no heat engine activity. This is used in
    cost calculation. Note `t_init` is the temperature in the last time-slot of last planning
    window, *not* the first time-slot of this planning window.
    '''
    t_base = base_soc(t_init, s=sustainment, l=len(self)) + soc(t_external, s=sustainment, e=(1-sustainment))
    return t_base

  def to_dict(self):
    data = super().to_dict()
    data.update({
      'sustainment': self.sustainment,
      'efficiency': self.efficiency,
      't_init': self.t_init,
      't_optimal': self.t_optimal,
      't_range': self.t_range,
      't_external': self.t_external
    })
    return data
