'''
This module includes convenience classes for doing projection onto *simple* convex polytopes.
The Intersection convex region uses dykstra's algorithm to finds the projection onto the
intersection of two regions. But it's slow and should not be used recursively. Recommended approach
is to formulate projection as a QP problem and use SciPy's minimize().
'''
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..typings import Number


class ConvexRegion(ABC):
  ''' Abstract base class for convex regions '''
  tol = 1e-10

  @abstractmethod
  def __len__(self) -> int:
    pass

  def is_in(self, point: NDArray[np.number]) -> np.bool:
    return (np.abs(self.project(point) - point) <= self.tol).all()

  @abstractmethod
  def project(self, point: NDArray[np.number]) -> NDArray[np.number]:
    pass


class HyperCube(ConvexRegion):
  _cube: NDArray[np.number]

  def __init__(self, bounds: Collection[Number]):
    self._cube = np.array(bounds)
    if len(self._cube.shape) != 2 or self._cube.shape[1] != 2:
      raise ValueError("Bad shape")

  def __len__(self):
    return len(self._cube)

  def project(self, point: Collection[Number]) -> NDArray[np.number]:
    if len(point) != len(self):
      raise ValueError("Point has wrong length (%d != %d)" % (len(point), len(self)))
    return np.array([max(self._cube[i][0], min(self._cube[i][1], p)) for i, p in enumerate(point)])

  @property
  def cube(self):
    return self._cube


class HalfSpace(ConvexRegion):
  ''' A closed half space of form x*normal <> offset, where <> is determined by sign parameter. '''
  _normal: NDArray[np.number]
  _offset: NDArray[np.number]
  _sign: int

  def __init__(self, normal: NDArray[np.number], offset: NDArray[np.number], sign: int):
    if sign == 0:
      raise ValueError("Sign must +ve or -ve")
    self._normal = np.array(normal)/np.linalg.norm(normal)
    self._offset = offset/np.linalg.norm(normal)
    self._sign = sign

  def __len__(self):
    return len(self._normal)

  def __str__(self):
    sign = "<=" if self._sign < 0 else ">="
    return "%s * x %s %f" % (self._normal, sign, self._offset)

  def project(self, point: NDArray[np.number]) -> NDArray[np.number]:
    ''' Move from point in the direction of normal until normal * point <> offset is satisfied '''
    if len(point) != len(self):
      raise ValueError("Point has wrong length (%d != %d)" % (len(point), len(self)))
    point = np.array(point)
    if self._sign > 0 and (point*self._normal).sum() < self._offset:
      diff = self._offset - self._normal.dot(point)
      point = point + self._normal*diff
    elif self._sign < 0 and (point*self._normal).sum() > self._offset:
      diff = self._offset - self._normal.dot(point)
      point = point + self._normal*diff
    return point

  @property
  def normal(self):
    return self._normal

  @property
  def offset(self):
    return self._offset

  @property
  def sign(self):
    return self._sign


class Slice(ConvexRegion):
  ''' An upper+lower parallel hyperplanes. For any given point only one projection is possible. '''
  _low: HalfSpace
  _high: HalfSpace

  def __init__(self, normal: NDArray[np.number], low: NDArray[np.number], high: NDArray[np.number]):
    if low > high:
      raise ValueError("Low offset (%f) must be =< high offset (%f)", (low, high))
    self._low = HalfSpace(normal, low, 1)
    self._high = HalfSpace(normal, high, -1)

  def __len__(self) -> int:
    return len(self._low)

  def __str__(self):
    return "%s AND %s" % (str(self._low), str(self._high))

  def project(self, point: NDArray[np.number]) -> NDArray[np.number]:
    if self._low.is_in(point):
      return self._high.project(point)
    else:
      return self._low.project(point)

  def is_in(self, point: NDArray[np.number]):
    return self._low.is_in(point) and self._high.is_in(point)


class Intersection(ConvexRegion):
  ''' A ConvexRegion region that is the intersection of two ConvexRegions '''
  _a: ConvexRegion
  _b: ConvexRegion
  _maxiter = 1e3

  def __init__(self, a: ConvexRegion, b: ConvexRegion):
    if not isinstance(a, ConvexRegion):  # type: ignore
      raise ValueError('Intersection must be of two existing ConvexRegion type. Given type %s', (type(a),))
    if not isinstance(b, ConvexRegion):  # type: ignore
      raise ValueError('Intersection must be of two existing ConvexRegion type. Given type %s', (type(a),))
    if len(a) != len(b):
      raise ValueError('Convex regions must have same dimensionality')
    self._a = a
    self._b = b

  def __len__(self):
    return len(self._a)

  def __str__(self):
    return "%s AND %s" % (str(self._a), str(self._a))

  def project(self, point: NDArray[np.number]) -> NDArray[np.number]:
    proj = self._a.project(point)
    if self._b.is_in(proj):
      return proj
    proj = self._b.project(point)
    if self._a.is_in(proj):
      return proj
    return self.dykstra_project(point)

  def is_in(self, point: NDArray[np.number]) -> np.bool:
    return self._a.is_in(point) and self._b.is_in(point)

  def dykstra_project(self, point: NDArray[np.number]) -> NDArray[np.number]:
    ''' See https://en.wikipedia.org/wiki/Dykstra's_projection_algorithm '''
    x = np.array(point)
    y = p = q = c = 0
    while isinstance(y, int) or c < self._maxiter and not (self._a.is_in(y) and self._b.is_in(y)):
      y = self._a.project(x + p)
      p = x + p - y
      x = self._b.project(y + q)
      q = y + q - x
      c += 1
    if c == self._maxiter:
      raise Exception("Dykstra's reached maxiter (%d) without converging within tol" % (self._maxiter))
    return x


class List(ConvexRegion):
  ''' Region is formed by a list of convex regions over disjoint subdomains of this region.
  The input point is treated as a matrix. It must have matching shape.
  '''
  _regions: Sequence[ConvexRegion]
  _axis: int = 0
  _shape: tuple[int, int]

  def __init__(self, regions: Sequence[ConvexRegion], axis: int = 0):
    if axis not in (0, 1):
      raise ValueError("axis must be 0|1. Given %d" % (axis,))
    l = len(regions[0])
    r = len(regions)
    self._axis = axis
    self._regions = regions
    self._shape = (r, l)if axis == 0 else (l, r)

  def __str__(self):
    _str = ""
    for r in self._regions:
      _str += str(r) + '\n'
    return _str

  def __len__(self) -> int:
    return self._shape[0]*self._shape[1]

  def project(self, point: NDArray[np.number]):
    point = np.array(point)
    if point.shape != self._shape:
      raise ValueError("Wrong shape. Given %s, require %s" % (str(point.shape), str(self._shape)))
    for i, r in enumerate(self._regions):
      if self._axis == 0:
        point[i, :] = r.project(point[i, :])
      else:
        point[:, i] = r.project(point[:, i])
    return point
