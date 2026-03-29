'''
Single Device and Generator with quadratic cost curve 24x1H day. The Device has no preference for
energy itself, however must consume at least 12kWh at any time of the day. Max rating is 3kW, thus
the minimum time window in which to operate is 4H.

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
  'title': 'The CDevice2 should want to consume at it\'s cbound limit because the marginal cost is less than the utility cost at max cbound',
}
basis = 48
demand_bounds = (0, 2)
supply_bounds = (-3, 0)


def make_deviceset():
  return DeviceSet('nw', [
    CDevice2(
      'demand',
      basis,
      demand_bounds,
      cbounds=(0, 10),
      p_l=-1.4,
      p_h=-0
    ),
    GDevice(
      'supply',
      basis,
      supply_bounds,
      cost_coeffs=[1, 1, 0]
    )
  ])

# This should be identical ..
# def make_deviceset():
#   return DeviceSet('nw', [
#     ADevice(
#       'demand',
#       basis,
#       demand_bounds,
#       cbounds=(0,10),
#       f=HLQuadraticCost(-1.4, 0, 0, 10)
#     ),
#     ADevice(
#       'supply',
#       basis,
#       supply_bounds,
#       f=ReflectedFunction(Poly1D(np.poly1d([1,1,0])))
#     )
#   ])

# # As should this
# def make_deviceset():
#   return DeviceSet('nw', [
#     ADevice(
#       'demand',
#       basis,
#       demand_bounds,
#       cbounds=(0,10),
#       f=HLQuadraticCost(-1.4, 0, 0, 10)
#     ),
#     ADevice(
#       'supply',
#       basis,
#       supply_bounds,
#       f=ReflectedFunction(Poly2D(np.tile([1,1,0], basis).reshape(basis, 3)))
#     )
#   ])
