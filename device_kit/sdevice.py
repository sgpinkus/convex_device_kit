import numdifftools as nd
import numpy as np

from .device import Device
from .utils import base_soc, soc, sustainment_matrix


class SDevice(Device):
  ''' Storage device. Behaviour is determined by settings but typically storage device draws and supplies resource.
  The cost function has a cost and a reward component:

    C(q(t)) = D(q(t)) + p*q(t)

  Since q(t) can be -ve for a storage device it's always preferable for the device to discharge -
  assuming the price is always +ve. The higher the price the more preferable it is. The cost has 3
  components that model:

    - The cost (price) of fast charging or discharging rates.
    - The cost (price) from a large number of charge discharge cycles over a day.
    - The cost (price) from deep discharge cycles.

  There are seven parameters to SDevice:

    - c1,c2,c3: coefficients for the first, second, third cost terms listed above.
    - capacity: the storage capacity of the device.
    - damage_depth: how deep is a deep discharge.
    - start: how much charge the device has at the start of the planning window as a ratio of capacity.
    - reserve: how much charge must the device be storing at the end of the planning window as a ratio of capacity.
    - efficiency: The *single* trip efficiency factor - [0,1]. Applied symmetrically to in/out flow.
        This means the round-trip efficiency (RTEF) is 1/2 this: efficiency = 1-(1-RTEF)/2. Ex, if
        the RTEF=0.9 then efficiency should be set to 1-(1-0.9)/2 = 0.95.
    - rate_clip: <rate_clip_rod, rate_clip_roc>: Must be None or >=1. If rate_clip_rod is 1 RoD is proportional to SoC/capacity, and
        if rate_clip_roc is 1 RoC is porportional to (capacity-SoC)/capacity. When the value is >1 the clipping kicks in
        later but the clipping rate is propotional to the value. Trapezoid ..

  '''
  _c1 = 0.0
  _c2 = 0.0
  _c3 = 0.0
  _capacity = 10
  _damage_depth = 0.0
  _start = 0.0
  _reserve = 0.0
  _efficiency = 1.0
  _sustainment = 1.0
  _sustainment_matrix = None
  _rate_clip = (None, None)

  def __init__(self, id, length, bounds, cbounds=None, **kwargs):
    super().__init__(id, length, bounds, cbounds=None, **kwargs)
    self._sustainment_matrix = sustainment_matrix(self.sustainment, len(self))

  def slice(self, history):
    '''Condition on observed history: advance start SoC to observed terminal SoC,
    adjust bounds and cbounds for remaining slots.'''
    from device_kit.utils import adjust_cbounds as _adjust_cbounds
    from device_kit.utils import base_soc, soc
    from device_kit.utils import sustainment_matrix as sm_fn
    history = np.asarray(history).reshape(1, -1)
    T = history.shape[1]
    if T >= len(self):
      raise ValueError(f'History length {T} must be less than device length {len(self)}')
    h1d = history[0]
    # Compute terminal SoC using just the history length — avoids charge_at()'s
    # assumption that the flow vector has the same length as the full device.
    terminal_soc = (
      base_soc(self.base(), s=self.sustainment, l=T)[-1]
      + soc(h1d, s=self.sustainment, e=self.efficiency)[-1]
    )
    if terminal_soc < 0 or terminal_soc > self.capacity:
      raise ValueError(
        f'History drives SDevice "{self.id}" SoC to {terminal_soc:.4f}, '
        f'which is outside [0, {self.capacity}]'
      )
    new_start = terminal_soc / self.capacity
    data = self.to_dict()
    data['length'] = len(self) - T
    data['bounds'] = self.bounds[T:]
    data['cbounds'] = _adjust_cbounds(self.cbounds, h1d, T, len(self))
    data['start'] = new_start
    sliced = self.__class__(**{k: v for k, v in data.items() if k in self._keys})
    sliced._sustainment_matrix = sm_fn(sliced.sustainment, len(sliced))
    return sliced

  def costv(self, s, p):
    return self.charge_costs(s) + s*p

  def cost(self, s, p):
    return self.costv(s, p).sum()

  def deriv(self, s, p):
    return self.charge_costs_deriv(s) + p

  def hess(self, s, p=0):
    ''' Return hessian approximation. '''
    return nd.Hessian(lambda x: self.cost(x, 0))(s.reshape(len(self)))

  def charge_costs(self, r):
    ''' Get total costs for flow vector `r`. '''
    r = r.reshape((len(self),))
    cost1 = (self.c1*r**2)
    cost2 = self.flip_cost_at(r)
    cost3 = self.deep_damage_at(r)
    return cost1 + cost2 + cost3

  def charge_costs_deriv(self, r):
    ''' Deriv of charge_costs(). '''
    r = r.reshape((len(self),))
    cost1_deriv = self.c1*2*r
    cost2_deriv = self.c2*-1*np.hstack((r[1:], [r[len(r)-1]]))
    cost3_deriv = self.deep_damage_at_deriv(r)
    return cost1_deriv + cost2_deriv + cost3_deriv

  def flip_cost_at(self, r):
    ''' Calculate total cost for flippyness in flow vector `r`. '''
    r = r.reshape((len(self),))
    return self.c2*-1*np.array([r[i]*r[i+1] for i in range(0, len(r)-1)] + [0])

  def deep_damage_at(self, r):
    ''' Calculate total cost for deep discharge in flow vector `r`. '''
    return self.c3*np.minimum((self.charge_at(r) - self.capacity*self.damage_depth), 0)**2

  def deep_damage_at_deriv(self, r):
    ''' Partial dervis of deep_damage_at(). Chain rule with outer minimum() and charge_at() wrt `r`. '''
    d = np.zeros(len(r))
    c = self.c3*2*np.minimum((self.charge_at(r) - self.capacity*self.damage_depth), 0)
    for i in range(0, len(c)):
      d += c[i]*np.hstack((np.ones(i+1), np.zeros(len(c)-i-1)))
    return d

  def charge_at(self, r):
    ''' Return SoC vector given the RoC vector `r`. No check is made to ensure `r` is feasible. '''
    return base_soc(self.base(), s=self.sustainment, l=len(self)) + soc(r, s=self.sustainment, e=self.efficiency)

  def charge_at_lossless(self, r):
    ''' Return SoC vector given the RoC vector `r`. No check is made to ensure r is feasible. '''
    return self.base()+r.cumsum()

  def base(self):
    return self.start*self.capacity

  @property
  def c1(self):
    return self._c1

  @property
  def c2(self):
    return self._c2

  @property
  def c3(self):
    return self._c3

  @property
  def capacity(self):
    return self._capacity

  @property
  def start(self):
    return self._start

  @property
  def reserve(self):
    return self._reserve

  @property
  def damage_depth(self):
    return self._damage_depth

  @property
  def efficiency(self):
    return self._efficiency

  @property
  def sustainment(self):
    return self._sustainment

  @property
  def rate_clip(self):
    return self._rate_clip

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this device.
    Constraints ensure at every time-slot the current storage level is between 0 and `capacity`.
    At any point level is determined only by initial level and sum of consumption upto that point.
    Also impose that at EOD level is at least reserve.
    '''
    constraints = Device.constraints.fget(self)
    reserve = self.capacity*self.reserve
    base = self.base()
    e = self.efficiency
    s = self.sustainment
    sustainment_matrix = self._sustainment_matrix
    def soc(r, i):
      mask = sustainment_matrix[i]
      return base*(s**(i+1)) + ((e**np.sign(r))*r).dot(mask)
    # Discrete integral always within [0,capacity]
    for i in range(0, len(self)):
      mask = sustainment_matrix[i]
      constraints += [
        # SoC >=0
        {
          'type': 'ineq',
          'fun': lambda r, i=i: soc(r, i),
          'jac': lambda r, mask=mask: (e**np.sign(r))*mask
        },
        # SoC <= capacity
        {
          'type': 'ineq',
          'fun': lambda r, i=i: self.capacity - soc(r, i),
          'jac': lambda r, mask=mask: -1*(e**np.sign(r))*mask
        }
      ]
    # RoC varies linearly as SoC varies capacity. rate_clip must be >=1
    if self.rate_clip[0]:
      for i in range(0, len(self)):
        mask = sustainment_matrix[i]
        constraints += [
          # RoC >= Rate_Clip*Min_RoC*((SoC)/capacity)
          {
            'type': 'ineq',
            'fun': lambda r, i=i: r.reshape(len(self))[i] - self.rate_clip[0]*self.lbounds[i]*((soc(r, i))/self.capacity),
            # 'jac': lambda r, mask=mask: (self.rate_clip[0]*self.lbounds[i]/self.capacity)*((e**np.sign(r))*mask)
          },
        ]
    if self.rate_clip[1]:
      for i in range(0, len(self)):
        mask = sustainment_matrix[i]
        constraints += [
          # RoC <= Rate_Clip*Max_RoC*((capacity-SoC)/capacity)
          {
            'type': 'ineq',
            'fun': lambda r, i=i: self.rate_clip[1]*self.hbounds[i]*((1 - soc(r, i)/self.capacity)) - r.reshape(len(self))[i],
            # 'jac': lambda r, mask=mask: (-1*self.rate_clip[1]*self.hbounds[i]/self.capacity)*((e**np.sign(r))*mask)
          },
        ]
    # At least reserve left at end of window.
    constraints += [
      {
        'type': 'ineq',
        'fun': lambda r: soc(r, len(self)-1) - reserve,
        'jac': lambda r, mask=sustainment_matrix[len(self)-1]: (e**np.sign(r))*mask
      },
    ]
    return constraints

  @c1.setter
  def c1(self, c1):
    if c1 < 0:
      raise ValueError('cost coefficients must be non -ve')
    if c1 <= self.c2 and self.c2 > 0:
      raise ValueError('c1 (%f) must be greater than c2 (%f)' % (c1, self.c2))
    self._c1 = c1

  @c2.setter
  def c2(self, c2):
    if c2 < 0:
      raise ValueError('cost coefficients must be non -ve')
    if c2 > self.c1 and self.c1 > 0:
      raise ValueError('c2 (%f) must be less than c1 (%f)' % (c2, self.c1))
    self._c2 = c2

  @c3.setter
  def c3(self, c3):
    if c3 < 0:
      raise ValueError('cost coefficients must be non -ve')
    self._c3 = c3

  @capacity.setter
  def capacity(self, capacity):
    if capacity <= 0:
      raise ValueError('capacity must be > 0')
    self._capacity = capacity

  @start.setter
  def start(self, start):
    if not 0 <= start <= 1:
      raise ValueError('start must be in [0,1]')
    self._start = start

  @reserve.setter
  def reserve(self, reserve):
    if not 0 <= reserve <= 1:
      raise ValueError('reserve must be in [0,1]')
    self._reserve = reserve

  @damage_depth.setter
  def damage_depth(self, damage_depth):
    if not 0 <= damage_depth <= 1:
      raise ValueError('damage_depth must be in [0,1]')
    self._damage_depth = damage_depth

  @efficiency.setter
  def efficiency(self, efficiency):
    if not 0 < efficiency <= 1.0:
      raise ValueError('efficiency factor must be in range (0, 1]')
    self._efficiency = efficiency

  @sustainment.setter
  def sustainment(self, sustainment):
    if not 0 < sustainment <= 1.0:
      raise ValueError('sustainment must be in range (0, 1]')
    self._sustainment = sustainment
    self._sustainment_matrix = sustainment_matrix(sustainment, len(self))

  @rate_clip.setter
  def rate_clip(self, rate_clip):
    try:
      rate_clip[0], rate_clip[1]
    except TypeError:
      rate_clip = (rate_clip, rate_clip)
    if rate_clip[0] is not None and not rate_clip[0] >= 1.0:
      raise ValueError('rate_clip must be >=1 or None')
    if rate_clip[1] is not None and not rate_clip[1] >= 1.0:
      raise ValueError('rate_clip must be >=1 or None')
    self._rate_clip = rate_clip
