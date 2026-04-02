import logging
from typing import Any, cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (
  OptimizeResult,
  minimize,
)

from device_kit.basedevice import BaseDevice
from device_kit.utils import project

logger = logging.getLogger(__name__)


def step(device, p, s, stepsize: int = 1, solver_options={}):
  ''' Take one step towards optimal demand for price vector `p`, using stepsize plus limited
  minimization. This means stepsize sets the upper bound of change in x, but stepsize can be large
  since limited minimization limits severity of overshoots.
  '''
  # Step & Projection
  s_next = s + stepsize * device.deriv(s, p)
  (s_next, o) = project(s_next, s, device.bounds, device.constraints)
  if not o.success:
    if o.status == 8:
      logger.warn(o)
    else:
      raise OptimizationException(o)
  # Limited minimization
  ol = minimize(
    lambda x, p=p: device.cost(s + x*(s_next - s), p),
    0.,
    method='SLSQP',
    bounds=[(0, 1)],
    options=solver_options,
  )
  if not ol.success:
    if ol.status == 8:
      logger.warn(ol)
    else:
      raise OptimizationException(ol)
  s_next = (s + ol.x*(s_next - s)).reshape(device.shape)
  return (s_next, ol)


def solve(
    device: BaseDevice,
    p: float | NDArray[np.number] = 0.0,
    s0: NDArray[np.number] | None = None,
    solver_options: dict[str, Any] = {},
    prox: float | None = None,
    cb=None,
    history: NDArray[np.number] | None = None,
) -> tuple[NDArray[np.number], OptimizeResult]:
  ''' Find the optimal demand for price for the given device and return it. Works on any device
  since only requires s and device.deriv. This method does not modify the device.

  Note AFAIK scipy.optimize only provides two methods that support constraints:
    - COBYLA (Constrained Optimization BY Linear Approximation)
    - SLSQP (Sequential Least SQuares Programming)
  Only SLSQP supports eq constraints. SLSQP is based on the Han-Powell quasi-Newton method. Apparently
  it uses some quadratic approximation, and the same method seems to be sometimes referred called
  SLS-Quadratic-Programming. This does not mean it is limited to quadratics. It should work with *any*
  convex function over a convex set.

  If `history` is provided (shape (device.shape[0], T) or (T,) for atomic devices), the device is
  first conditioned on the observed history via device.slice(history), reducing the problem to slots
  [T:]. `p` and `s0`, if provided as arrays, are sliced to [T:] automatically. The returned flow
  matrix covers only the remaining slots; prepend history to reconstruct the full schedule:

      x_remaining, meta = solve(device, p=p, history=history)
      x_full = np.concatenate([history, x_remaining], axis=1)

  @see http://www.pyopt.org/reference/optimizers.slsqp.html
  @see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
  '''
  if history is not None:
    history = np.asarray(history)
    if history.ndim == 1:
      history = history.reshape(1, -1)
    T = history.shape[1]
    device = device.slice(history)
    if np.asarray(p).ndim > 0:
      p = np.asarray(p)[..., T:]
    if s0 is not None:
      # s0 may be flat (n_rows*n_cols) or shaped; slice off first T cols per row
      s0_arr = np.asarray(s0).reshape(device.shape[0], -1)
      s0 = s0_arr[:, T:].flatten()

  _solver_options: dict[str, Any] = {'ftol': 1e-6, 'maxiter': 1000, 'disp': False}
  _solver_options.update(solver_options)
  logger.debug({'prox': prox, '_solver_options': solver_options, 'cb': cb})  # type: ignore

  # Don't attempt solve if input constrained to single solution.
  if (device.bounds[:, 0] == device.bounds[:, 1]).all():
    warn("Solution is fixed.")

  # Find a (assumed) feasible starting point
  s0 = (s0 if s0 is not None else device.project(np.zeros(device.shape))).flatten()

  args: dict[str, Any] = {
    'fun': lambda s, p=p: device.cost(s, p),
    'x0': s0,
    'jac': lambda s, p=p: device.deriv(s, p),
    'method': 'SLSQP',
    'bounds': device.bounds,
    'constraints': device.constraints,
    'options': _solver_options,
  }
  if cb:
    args.update({
      'callback': lambda x: cb(device, x)
    })
  if prox:
    args.update({
      'fun': lambda s, p=p: device.cost(s, p) + (1/(2*prox))*((s-s0)**2).sum(),
      'jac': lambda s, p=p: device.deriv(s, p).flatten() + (1/prox)*(s-s0),
    })
  o: MinimizeResult = cast(MinimizeResult, minimize(**args))
  if not o.success:
    raise OptimizationException(o)
  return ((o.x).reshape(device.shape), o)


class OptDebugCb():
  ''' Convenience minimize callback '''

  def __init__(self):
    self.i = 0

  def __call__(self, device: BaseDevice, x: NDArray[np.number]):
    logger.debug('step=%d; cost=%.6f' % (self.i, device.cost(x, 0)))
    self.i += 1


class OptimizationException(Exception):  # pragma: no cover
  ''' Some type of optimization error. '''
  o: Any = None  # An optional optimization method specific status report (i.e. OptimizeResult).

  def __init__(self, *args: list[Any]):
    self.o = args[0] if args else None
    super(Exception, self).__init__(*args)


class MinimizeResult(OptimizeResult):
  success: bool
  x: np.ndarray
