import functools
from copy import deepcopy
from functools import reduce

import numpy as np
from scipy.optimize import minimize


def base_soc(b, s, l):
  ''' Apply decay to scalar b over times l, at rate 1-s '''
  return b*(s**np.arange(1, l+1))


def soc(r, s, e):
  ''' Get "state of charge" or rather state of something. This is basically calculating a discrete
  integral for all values between [0,len(r)] given r, some sustainment `s` and efficiency `e` factors.
  '''
  r = np.array(r)
  if len(r.shape) != 1:
    raise ValueError('Input value must have vector shape not %s' % (r.shape,))
  sm = sustainment_matrix(s, len(r))
  return ((r*(e**np.sign(r)))*sm).cumsum(axis=1).diagonal()


@functools.lru_cache()
def sustainment_matrix(s, l):
  ''' Returns a matrix with coefficients for basically thermal-ish decay. Note "sustainment" is
  the opposite of decay, sustainment (s) or 1 means zero loss.
  '''
  if s == 1:
    return np.tril(np.ones((l, l)))
  return np.tril(s**power_matrix(l))


@functools.lru_cache()
def power_matrix(l):
  ''' Returns a lower triangular matrix, that can be used in a power series. '''
  return np.array([i.cumsum() for i in np.triu(np.ones((l, l)), 1)]).transpose()


def care2bounds(device):
  ''' The bounds style used by Device is same a scipy minimize, but it's annoying. This function
  converts `care` array, plus `bounds` 2-tuple to Device.bounds style bounds.
  '''
  device = deepcopy(device)
  care = device['care']
  bounds = device['bounds']
  del device['care']
  if len(bounds) == 2:
    device['bounds'] = np.stack((care*bounds[0], care*bounds[1]), axis=1)
  else:  # Assume bounds is a vector
    device['bounds'] = np.stack((care*bounds, care*bounds), axis=1)
  return device


def on2bounds(device, l):
  ''' Convert on, and bounds pair to powermarket style bounds '''
  device = deepcopy(device)
  on = device['on']
  bounds = device['bounds']
  on_vector = np.zeros(l)
  del device['on']
  for i in range(0, len(on), 2):
    on_vector[on[i]:on[i+1]+1] = 1
  if len(bounds) == 2:
    device['bounds'] = np.stack((on_vector*bounds[0], on_vector*bounds[1]), axis=1)
  else:  # Assume bounds is a vector
    device['bounds'] = np.stack((on_vector*bounds, on_vector*bounds), axis=1)
  return device


def zmm(x, keep, axis=0, fn=None):
  ''' Zero mask out rows/cols along axis not in keep index, applying fn(<kept>) if fn is provided. '''
  r = np.zeros(x.shape)
  if axis == 0:
    i = x[keep, :]
    r[keep, :] = fn(i).reshape(i.shape) if fn else i
  elif axis == 1:
    i = x[:, keep]
    r[:, keep] = fn(i).reshape(i.shape) if fn else i
  else:
    raise np.AxisError(axis)
  return r


def project(p, x0, bounds=[], constraints=[], solver_options={}):
  ''' Find the point in feasible region closest to p. '''
  p = p.flatten()
  options = {
    'ftol': 1e-9,
    'disp': False,
    'maxiter': 200
  }
  options.update(solver_options)
  o = minimize(lambda s, p=p: ((s - p)**2).sum(), x0, method='SLSQP',
               jac=lambda s, p=p: 2*(s - p),
               options=options,
               bounds=bounds,
               constraints=constraints
  )
  return (o.x.reshape(x0.shape), o)


def flatten(x):
  return reduce(lambda a, b: list(a) + list(b), x, [])


def get_device_by_id(deviceset, id):
  try:
    return list(filter(lambda v: v.id == id, deviceset.devices))[0]
  except BaseException:
    return None
