import re
from numbers import Number
from typing import Any, Sequence
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from .basedevice import BaseDevice
from .projection import ConvexRegion, HyperCube

BoundsInput = Number | Sequence[Number] | Sequence[Sequence[Number]]

CBound = tuple[float, float, int, int]
CBoundsInput = Sequence[CBound] | tuple[Number, Number]


class Device(BaseDevice):
  ''' BaseDevice implementation for single Device. '''
  _id: str = ''                  # The identifier of this device.
  _len: int = 0                  # Fixed length of the following vectors / the planning window.
  _bounds: NDArray[np.float64]   # Vector of 2-tuple min/max bounds on r.
  _cbounds: list[CBound] | None  # list of 4-Tuple cummulative min/max bounds. Cummulative bounds are optional.
  _feasible_region: ConvexRegion    # Convex region representing *only* bounds and cbounds. Convenience.
  _keys: list[str]

  def __init__(self, id: str, length: int, bounds: Any, cbounds: CBoundsInput | None = None, **meta: dict[str, Any]):  # type: ignore
    ''' Validate and set field. Build an incomplete feasible_region for convenience sake. Class allows
    any arbitrary field to be passed which are set on the object and add to field that will be
    serialized with an instance (fields that are set on the object after init are not serialized. This
    is by design). Sub-classes may choose to override this initializer to provide complex object
    construction. In this case they must add additional keys to _keys or also override the to_dict()
    method. Alternatively they may just define setters which will be called for all keys in **meta.
    '''
    if not isinstance(id, str) or not re.match('(?i)^[a-z0-9][a-z0-9()\\[\\]\\+_-]*$', id):  # type: ignore
      warn('id should be a non empty string matching "^(?i)[a-z0-9][a-z0-9_-]*$" not "%s"' % (id,))
    self._length = length
    self._id = id
    self.bounds = bounds
    self.cbounds = cbounds
    self._feasible_region = HyperCube(self.bounds)
    self._keys = ['id', 'length', 'bounds', 'cbounds']
    for k, v in meta.items():
      setattr(self, k, v)
    self._keys += list(meta.keys())

  def __str__(self) -> str:
    ''' Print main settings. Dont print the actual min/max bounds vectors because its too verbose. '''
    _str = 'type=%s; id=%s; length=%d; bounds_bounds=%.3f/%.3f; cbounds=%s' % (
      self.__class__.__name__,
      self.id,
      len(self),
      self.lbounds.min(),
      self.hbounds.max(),
      self.cbounds
    )
    _str = '; '.join([_str] + ['{k}={v}'.format(k=k, v=getattr(self, k)) for k in self._keys if k not in ['id', 'length', 'bounds', 'cbounds']])  # type: ignore
    return _str

  def __len__(self) -> int:
    return self._length

  def cost(self, s, p) -> float:  # type: ignore
    ''' Get scalar cost (inverse utility) value for `s` consumption, at price (parameter) `p`. This base Device's
    cost function makes an assumption device cares linearly about costs. Generally all sub devices
    should do this too.
    '''
    return float((s*np.asarray(p)).sum())

  def deriv(self, s, p):  # type: ignore
    ''' Get jacobian vector of the cost at `s`, at price `p`, which is just -p. '''
    return p*np.ones(len(self))

  def hess(self, s, p=0):  # type: ignore
    ''' Get hessian vector of the cost at `s`, at price `p`. With linear cost for the numeriare
    price drops out.
    '''
    return np.zeros((len(self), len(self)))

  @property
  def id(self):
    return self._id

  @property
  def length(self):
    return self._length

  @property
  def shape(self) -> tuple[int, int]:
    return (1, len(self))

  @property
  def shapes(self):
    return np.array([self.shape])

  @property
  def partition(self):
    ''' Returns array of (offset, length) tuples for each sub-device's mapping onto this device's
    flow matrix.
    '''
    return np.array([[0, 1]])

  @property
  def bounds(self):
    return self._bounds

  @property
  def lbounds(self):
    return np.array(self.bounds[:, 0])

  @property
  def hbounds(self):
    return np.array(self.bounds[:, 1])

  @property
  def cbounds(self):
    return self._cbounds

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this device.
    Contraints at this level are just cbounds. bounds constraints are generally handled separately.
    '''
    constraints: list[Any] = []
    if self.cbounds:
      for cbound in self.cbounds:
        l, h, s, e = cbound  # type: ignore
        constraints += [{  # type: ignore
          'type': 'ineq',
          'fun': lambda s: s.dot(np.ones(len(self))) - l,  # type: ignore
          'jac': lambda s: np.ones(len(self))  # type: ignore
        },
        {
          'type': 'ineq',
          'fun': lambda s: h - s.dot(np.ones(len(self))),  # type: ignore
          'jac': lambda s: -1*np.ones(len(self))  # type: ignore
        }]
    return constraints

  @property
  def params(self):
    ''' ~BWC '''
    return self.to_dict()

  @bounds.setter
  def bounds(self, bounds: Any):  # type: ignore
    ''' See _validate_bounds() '''
    self._bounds = self.validate_bounds(bounds)
    self._feasible_region = HyperCube(self.bounds)

  @cbounds.setter
  def cbounds(self, cbounds: CBoundsInput | None) -> None:
    ''' Set cbounds ensuring they are feasible wrt (l|h)bounds. cbounds is either a 2-tuple or a list of 4-tuples. Each
    4-tuple is (lower_bound, upper_bound, start_index, end_index). cbounds are always stored in 4-tuple form.
    The actual range is [s,e) just like a python range, so to be contiguous e_{i} == s_{i+1}, although continuity isn't
    checked here.
    '''
    def set_cbound(cbound: CBound):
      assert self._cbounds is not None
      if not hasattr(cbound, '__len__') or len(cbound) != 4:
        raise ValueError(f'cbound must be a 4-tuple not "{cbound}"')
      if cbound[1] <= cbound[0]:
        raise ValueError('max cbound (%f) must be > min cbound (%f)' % (cbound[1], cbound[0]))
      if self.lbounds[cbound[2]:cbound[3]].sum() > cbound[1]:
        raise ValueError('cbounds infeasible; min possible sum (%f) is > max cbounds (%f)' % (self.lbounds[cbound[2]:cbound[3]].sum(), cbound[1]))
      if self.hbounds[cbound[2]:cbound[3]].sum() < cbound[0]:
        raise ValueError('cbounds infeasible; max possible sum (%f) is < min cbounds (%f)' % (self.hbounds[cbound[2]:cbound[3]].sum(), cbound[0]))
      self._cbounds.append(cbound)
    self._cbounds = []
    if cbounds is None:
      self._cbounds = None
      return
    elif not hasattr(cbounds, '__len__'):
      raise ValueError('cbounds should be a list of 4-tuples or a 2-tuple')
    if len(cbounds) == 2 and not hasattr(cbounds[0], '__len__'):
      set_cbound(tuple(cbounds) + (0, len(self)))  # type: ignore
    else:
      for cbound in cbounds:
        set_cbound(cbound)  # type: ignore

  @params.setter
  def params(self, params: dict[str, Any]):
    ''' Convenience buld setter. '''
    for k, v in params.items():
      setattr(self, k, v)

  def project(self, s: NDArray[np.number]) -> NDArray[np.number]:
    return self._feasible_region.project(s.reshape(len(self))).reshape(self.shape)

  def slice(self, history: NDArray[np.number]) -> "Device":
    ''' Return a new Device of the same type covering slots [T:], conditioned on history.
    history must have shape (1, T) or (T,).  Subclasses with extra state (cost params,
    etc.) should override this and call super().slice() to get the adjusted base kwargs,
    then reconstruct with their own extra params.
    '''
    from device_kit.utils import adjust_cbounds as _adjust_cbounds
    history = np.asarray(history).reshape(1, -1)
    T = history.shape[1]
    if T >= len(self):
      raise ValueError(f'History length {T} must be less than device length {len(self)}')
    h1d = history[0]
    new_bounds = self.bounds[T:]
    new_cbounds = _adjust_cbounds(self.cbounds, h1d, T, len(self))
    data = self.to_dict()
    data['length'] = len(self) - T
    data['bounds'] = new_bounds
    data['cbounds'] = new_cbounds
    # Strip any keys that the constructor doesn't accept via **meta but that
    # to_dict() emits (none for Device itself, but subclasses may add some).
    return self.__class__(**{k: v for k, v in data.items() if k in self._keys})

  def to_dict(self) -> dict[str, Any]:
    ''' Dump object as dict. Dict should allow re-init of instance via cls(**data). See __init__(),
    from_dict().
    '''
    data = {k: getattr(self, k) for k in self._keys}
    return data
