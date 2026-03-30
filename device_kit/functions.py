'''
Various preference functions. Can be used with ADevice and else where.
See notes on Function.
'''
from abc import ABC, abstractmethod
from functools import reduce

import numdifftools as nd
import numpy as np


class Function(ABC):
  ''' Input x should typically be a vector with __call_, deriv, hess returning
  scalar, (N,), (N,N) shaped results. Some functions may accept scalars for all
  of __call_, deriv, hess and in that case return scalars, but this should be
  considered a special sub-type.
  NOTE/TODO: Some sub-types also happen to accept a scalar only for __call__() with same
  result as array of len 1 but won't accept a scalar for deriv, hess.
  '''

  @abstractmethod
  def __call__(self, x):
    pass

  @abstractmethod
  def deriv(self, x):
    pass

  @abstractmethod
  def hess(self, x):
    pass


class NullFunction(Function):
  ''' Default null function. f parameter has to ADevice has to look like this. '''

  def __call__(self, x):
    return 0

  def deriv(self, x):
    return 0

  def hess(self, x):
    return 0


class SumFunction(Function):
  ''' '''
  functions = []

  def __init__(self, functions: list[Function]) -> None:
    self.functions = functions if len(functions) else [NullFunction()]

  def __call__(self, x):
    return np.array([v(x) for v in self.functions]).sum()

  def __str__(self):
    return f'Sum({[str(f) for f in self.functions]})'

  def deriv(self, x):
    return np.array([v.deriv(x) for v in self.functions]).sum(axis=0)

  def hess(self, x):
    return np.array([v.hess(x) for v in self.functions]).sum(axis=0)


class ReflectedFunction(Function):
  function = None

  def __init__(self, function: Function) -> None:
    self.function = function

  def __call__(self, x):
    return self.function(-1*x)

  def __str__(self):
    return f'{str(self.function)} o -x'

  def deriv(self, x):
    return self.function.deriv(-1*x)*-1

  def hess(self, x):
    return self.function.hess(-1*x)


class Poly2D(Function):
  ''' Wraps up an vector of np.poly1ds. Makes it slightly easier to get derivs. Not when called with a vector np.poly1d returns
  a vector of the singular polynomial applied to each index of the input vector. But that means deriv is technically
  incorrect. Poly2D call return the scalar sum of the said vector to be consistent with other functions.
  '''
  coeffs = None
  _polys = None
  _deriv = None
  _hess = None

  def __init__(self, coeffs):
    ''' `c` should be an array of arrays. Each array the coeffs of a np.poly1d. '''
    self.coeffs = np.array(coeffs)
    self._polys = [np.poly1d(c) for c in self.coeffs]

  def __call__(self, x):
    return self.vector(x).sum()

  def __len__(self):
    return len(self.coeffs)

  def vector(self, x):
    return np.array([self._polys[k](v) for k, v in enumerate(np.array(x).reshape(len(self)))])

  def deriv(self, x):
    if not self._deriv:
      self._deriv = Poly2D([np.polyadd(np.zeros(len(c)), np.poly1d(c).deriv().coeffs) for c in self.coeffs])
    return self._deriv.vector(x)

  def hess(self, x):
    if not self._hess:
      self._hess = Poly2D([np.polyadd(np.zeros(len(c)), np.poly1d(c).deriv(2).coeffs) for c in self.coeffs])
    return np.diag(self._hess.vector(x))


class Poly2DOffset(Function):
  ''' @see Poly2D
  '''
  coeffs = None
  _polys = None
  _deriv = None
  _hess = None

  def __init__(self, coeffs):
    ''' `c` should be an array of arrays. Each array the coeffs of a np.poly1d. '''
    self.coeffs = np.array(coeffs)[:, 0:3]
    self.offsets = np.array(coeffs)[:, 3]
    self._polys = [np.poly1d(c) for c in self.coeffs]

  def __call__(self, x):
    return self.vector(x).sum()

  def __len__(self):
    return len(self.coeffs)

  def vector(self, x):
    return np.array([self._polys[k](v + self.offsets[k]) for k, v in enumerate(np.array(x).reshape(len(self)))])

  def deriv(self, x):
    if not self._deriv:
      _coeffs = [np.polyadd(np.zeros(len(c)), np.poly1d(c).deriv().coeffs) for c in self.coeffs]
      self._deriv = Poly2DOffset(np.concatenate((_coeffs, self.offsets.reshape((len(self), 1))), axis=1))
    return self._deriv.vector(x)

  def hess(self, x):
    if not self._hess:
      _coeffs = [np.polyadd(np.zeros(len(c)), np.poly1d(c).deriv(2).coeffs) for c in self.coeffs]
      self._hess = Poly2DOffset(np.concatenate((_coeffs, self.offsets.reshape((len(self), 1))), axis=1))
    return np.diag(self._hess.vector(x))


class Poly1D():
  def __init__(self, poly1d):
    self.f = poly1d

  def __call__(self, x):
    return self.f(x).sum()

  def deriv(self, x):
    return self.f.deriv()(x)

  def hess(self, x):
    return np.diag(self.f.deriv(2)(x))


class X2D(Function):
  ''' Apply X *scalar* functions to a vector input. Assumes function takes a scalar input. Function does not accept
  scalar input. '''
  functions = []

  def __init__(self, functions: Function) -> None:
    self.functions = functions

  def __len__(self):
    return len(self.functions)

  def __call__(self, x):
    return np.array([self.functions[k](v) for k, v in enumerate(np.array(x).reshape(len(self)))]).sum()

  def deriv(self, x):
    return np.array([self.functions[k].deriv(v) for k, v in enumerate(np.array(x).reshape(len(self)))]).reshape(-1)

  def hess(self, x):
    ''' Each hess value should be an 1x1 matrix '''
    return np.diag(np.array([self.functions[k].hess(v) for k, v in enumerate(np.array(x).reshape(len(self)))]).flatten())


class RangesFunction(Function):
  ''' Apply i-th function to the i-th range of the input vector. Assumes input is a vector and has length equal to that
  covered by the ranges. '''

  def __init__(self, range_functions: tuple[tuple[int, int], Function]) -> None:
    self.ranges = [f[0] for f in range_functions]
    self._validate_ranges(self.ranges)
    self.functions = [f[1] for f in range_functions]
    self._derivs = [f.deriv for f in self.functions]
    self._hessians = [f.hess for f in self.functions]
    self._len = self.ranges[-1][1]

  def __len__(self):
    return self._len

  def __call__(self, x):
    x = x.reshape((len(self),))
    return np.array([self.functions[k](x[range(*_range)]) for k, _range in enumerate(self.ranges)]).sum()

  def deriv(self, x):
    x = x.reshape((len(self),))
    range_derivs = [self.functions[k].deriv(x[range(*_range)]) for k, _range in enumerate(self.ranges)]
    return np.array(reduce(lambda a, b: list(a) + list(b), range_derivs, []))  # np.flatten() or "+" doesn't work with inhomogenous.

  def hess(self, x):
    x = x.reshape((len(self),))
    range_hessians = [self.functions[k].hess(x[range(*_range)]) for k, _range in enumerate(self.ranges)]
    y = np.zeros((len(self), len(self)))
    for i, m in zip([r[0] for r in self.ranges], range_hessians):
      y[i:i+m.shape[0], i:i+m.shape[1]] = m
    return y

  def _validate_ranges(self, ranges):
    if ranges[0][0] != 0:
      raise ValueError('ranges must start at zero')
    if not (np.array([v[0] - (ranges[i-1][1] if i > 0 else 0) for i, v in enumerate(ranges)]) == 0).all():
      raise ValueError(f'ranges must be contiguous and non overlapping not {ranges}')


class InnerSumFunction(Function):
  ''' This is a special simple case of composition, f o g - because deriv of x.sum() is just ones.
  TODO: Just implement a composition and the chain rule.
  NOTE: This function assumes the outer_function accepts a scalar for all of __call__(), deriv(), hess(), but
  this isn't the case for many functions ...
  '''
  def __init__(self, outer_function) -> None:
    self.outer_function = outer_function

  def __call__(self, x):
    return self.outer_function(np.array(x).sum())

  def deriv(self, x):
    return self.outer_function.deriv(np.array(x).sum())*np.ones(len(x))

  def hess(self, x):
    return self.outer_function.hess(np.array(x).sum())*np.eye(len(x))


class InformationEntropy():
  ''' Information entropy preference function. Entropy is bad.
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, c=1):
    self.c = c  # Scalar coefficient.

  def __call__(self, r):
    return self.c*InformationEntropy.info_entropy(r)

  def deriv(self, x):
    return nd.Jacobian(lambda x: self.c*InformationEntropy.info_entropy(x))(x)

  def hess(self, x):
    return nd.Hessian(lambda x: self.c*InformationEntropy.info_entropy(x))(x)

  @staticmethod
  def info_entropy(r):
    r = np.array([i for i in np.abs(r) if i])
    r = r/r.sum()
    return (r*np.log(r)).sum()


class TemporalVariance():
  ''' Equivalent to moment of inertia if time is distance and consumption is mass. Inertia is bad.
  @todo define deriv() and hess() numerically.
  '''

  def __init__(self, c=1):
    self.c = c  # Scalar coefficient.

  def __call__(self, r):
    return self.c*TemporalVariance.inertia(r)

  def deriv(self, x):
    return nd.Jacobian(lambda x: self.c*TemporalVariance.inertia(x))(x)

  def hess(self, x):
    return nd.Hessian(lambda x: self.c*TemporalVariance.inertia(x))(x)

  @staticmethod
  def inertia(r):
    t = np.arange(len(r))
    return (((t - TemporalVariance.com(r))**2)*r).sum()

  @staticmethod
  def com(r):
    ''' Center of Mass. Merely weighted avg of time-slots. '''
    return np.average(np.arange(len(r)), weights=r)


class CobbDouglas():
  ''' Cobb Douglas *with* constant returns to scale. '''

  def __init__(self, a, c=1):
    self.a = a  # Cobb Douglas coefficients.
    self.c = c  # Scalar coefficient.

  def __call__(self, r):
    return self.c*CobbDouglas.cobb_douglas(r, self.a)

  def deriv(self, x):
    return nd.Jacobian(lambda x: self.c*CobbDouglas.cobb_douglas(x, self.a))(x)

  def hess(self, x):
    return nd.Hessian(lambda x: self.c*CobbDouglas.cobb_douglas(x, self.a))(x)

  @staticmethod
  def cobb_douglas(r, a):
    return (r**(a/a.sum())).prod()


class ABCCost():
  ''' Provides a convex, and decreasing function described by 3 params; a,b,c.

  Consider the polynomial section between [0,1]:

    f(x) = x^b

  Let s(x) give x scaled from [x_max, x_min] -> [0,1]. Then  when `a` is zero this cost function is:

    g(x) = c*f(s(x)

  When a is not zero we get scaling [x_max, x_min] -> [a, 1]. This means when x = x_max the derivative
  is not zero which is often what you want.

  '''
  a = 0
  b = 2
  c = 1
  x_l = None
  x_h = None

  def __init__(self, a, b, c, x_l, x_h):
    [self.a, self.b, self.c, self.x_l, self.x_h] = a, b, c, x_l, x_h
    self._cost_fn = lambda x: np.vectorize(ABCCost._cost, otypes=[float])(x, self.a, self.b, self.c, self.x_l, self.x_h)
    self._deriv_fn = lambda x: np.vectorize(ABCCost._deriv, otypes=[float])(np.array(x).reshape(-1), self.a, self.b, self.c, self.x_l, self.x_h)
    self._hess_fn = lambda x: np.diag(np.vectorize(ABCCost._hess, otypes=[float])(np.array(x).reshape(-1), self.a, self.b, self.c, self.x_l, self.x_h))

  def __call__(self, x):
    return self._cost_fn(x).sum()

  def deriv(self, x):
    return self._deriv_fn(x)

  def hess(self, x):
    return self._hess_fn(x)

  @staticmethod
  def _cost(x, a, b, c, x_l, x_h):
    ''' The cost function on scalar. '''
    if x_l == x_h:
      return 0
    return c*ABCCost.q(x, x_l, x_h, a)**b

  @staticmethod
  def _deriv(x, a, b, c, x_l, x_h):
    ''' The derivative of cost function on scalar. '''
    if x_l == x_h:
      return 0
    return -c*b*ABCCost.q(x, x_l, x_h, a)**(b-1)

  @staticmethod
  def _hess(x, a, b, c, x_l, x_h):
    if x_l == x_h:
      return 0
    return c*(b-1)*ABCCost.q(x, x_l, x_h, a)**(b-2)

  @staticmethod
  def s(x, x_l, x_h):
    ''' scalce -> 0 as x -> x_h, -> 1 as x -> x_l '''
    return (x_h - x)/(x_h - x_l)

  @staticmethod
  def q(x, x_l, x_h, a):
    return (1 - ABCCost.s(x, x_l, x_h))*a + ABCCost.s(x, x_l, x_h)


class HLQuadraticCost():
  p_l = -1
  p_h = 0
  x_l = x_h = None
  _cost_fn = None
  _deriv_fn = None
  _hess_fn = None

  def __init__(self, p_l, p_h, x_l, x_h):
    [self.p_l, self.p_h, self.x_l, self.x_h] = p_l, p_h, x_l, x_h
    self._cost_fn = lambda x: np.vectorize(HLQuadraticCost._cost, otypes=[float])(x, self.p_l, self.p_h, self.x_l, self.x_h)
    self._deriv_fn = lambda x: np.vectorize(HLQuadraticCost._deriv, otypes=[float])(np.array(x).reshape(-1), self.p_l, self.p_h, self.x_l, self.x_h)
    self._hess_fn = lambda x: np.diag(np.vectorize(HLQuadraticCost._hess, otypes=[float])(np.array(x).reshape(-1), self.p_l, self.p_h, self.x_l, self.x_h))

  def __call__(self, r):
    return self._cost_fn(r).sum()

  def __str__(self):
    return f'p_l={self.p_l}, p_h={self.p_h}, x_l={self.x_l}, x_h={self.x_h}'

  def deriv(self, x):
    return self._deriv_fn(x)

  def hess(self, x):
    return self._hess_fn(x)

  @staticmethod
  def _cost(x, p_l, p_h, x_l, x_h):
    ''' The cost function on scalar. '''
    if x_l == x_h:
      return 0
    b = p_l
    a = (p_h - p_l)/2
    c = a*(-b/(2*a))**2 + b*(-b/(2*a))
    return (x_h - x_l)*np.poly1d([a, b, 0])((x - x_l)/(x_h - x_l)) - c*(x_h - x_l)

  @staticmethod
  def _deriv(x, p_l, p_h, x_l, x_h):
    ''' The derivative of cost function on a scalar. Returned valus is expansion of:
         np.poly1d([(p_h - p_l)/2, p_l, 0]).deriv(QuadraticCost12.scale(x, x_l, x_h))
    '''
    if x_l == x_h:
      return 0
    return (p_h - p_l)*((x - x_l)/(x_h - x_l)) + p_l

  @staticmethod
  def _hess(x, p_l, p_h, x_l, x_h):
    ''' The 2nd derivative of cost function on a scalar. '''
    if x_l == x_h:
      return 0
    return (p_h - p_l)/(x_h - x_l)


class DemandFunction(Function):
  ''' Not CD but convex if is convex '''

  def __init__(self, inner_function: np.poly1d) -> None:
    ''' f should be a scalar functional like poly1d '''
    self.inner_function = inner_function

  def __call__(self, x):
    return self.inner_function(np.max(x))

  def deriv(self, x):
    _x = np.zeros(np.array(x).shape)
    i = np.argmax(x)
    _x[i] = self.inner_function.deriv()(x[i])
    return _x

  def hess(self, x):
    _x = np.zeros(np.array(x).shape)
    i = np.argmax(x)
    _x[i] = self.inner_function.deriv(2)(x[i])
    return np.diag(_x)
