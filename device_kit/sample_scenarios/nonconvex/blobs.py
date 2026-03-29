'''

 - Single Device, and Generator.
 - Generator has quadratic cost curve but the 2nd order term decreases (cheaper) towards end of day.
 - 24x1H day.
 - The Device must consume at least 12kWh at any time of the day. Max rating is 3kW, thus the minimum time window in which to operate is 4H.

Given the quadratic cost curve ordinarily the optimal solution would be  to consume 0.5kW in every
hour. However, the device's utility function includes a factor to penalize dispersion from some
arbitrary mean. This should cause consumtion to lump together. The lumpiness is a result of a tension
between the quadratic costs and dispersion factor.
'''
import numpy as np
from device_kit import *
from device_kit.functions import *


np.set_printoptions(
  precision=6,
  linewidth=1e6,
  threshold=1e6,
  formatter={
    'float_kind': lambda v: '%0.6f' % (v,),
    'int_kind': lambda v: '%d' % (v,),
  },
)
meta = {
  'title': '''Penalizing time dispersion of load with fixed but time flexible energy req.
Load requires exactly 12kWh at any time between [04:00-19:00] '''
}
dimension = 24
phase = 0  # np.pi/4
cost = np.stack((
  2*np.sin(np.linspace(0+phase, np.pi+phase, dimension))+1,
  np.ones(dimension)*0.001,
  np.zeros(dimension)),
  axis=1
)


class BlobDevice(ADevice):
  ''' Overrides ADevice but constraint and f are not params. Only c is '''

  def __init__(self, id, length, bounds, cbounds=None, c=1):
    super().__init__(id, length, bounds, cbounds, c=c)
    self._f = TemporalVariance(self.c)


def make_deviceset():
  care = np.hstack((np.zeros(4), np.ones(16), np.zeros(4)))  # {18-23}
  bounds = np.stack((care*0, care*6), axis=1)
  return DeviceSet('nw', [
    BlobDevice(
      'demand',
      dimension,
      bounds,
      (12, 12),
      **{'c': 1}
    ),
    BlobDevice(
      'demand2',
      dimension,
      bounds,
      (12, 12),
      **{'c': 1}
    ),
    BlobDevice(
      'demand3',
      dimension,
      bounds,
      (12, 12),
      **{'c': 1}
    ),
    GDevice(
      'supply',
      dimension,
      np.stack((-20*np.ones(dimension), np.zeros(dimension)), axis=1),
      None,
      **{'cost_coeffs': cost}
    )
  ])


def matplot_network_writer_hook(event, fig, writer=None):
  if event != 'after-update':
    return
  p = Poly2D(cost)
  fig.axes[0].set_xlim(0, 30)
  fig.axes[0].plot(p.vector(np.ones(dimension)), label='quad_cost')
  fig.axes[0].set_ylim(0, 6)
