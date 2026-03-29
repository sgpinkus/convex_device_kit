'''
We have an EV charging station powered by PV and some kind of generator with a convex cost curve.
Users are assumed to be residence that live near by and need cars charged at regular intervals while
they are at home. t=0 is to 00:00. User are split into:

  - Early leavers. Need car charged by 5am. Car is available in the night slightly earlier.
  - Mid leavers. Need car charged by 8am.
  - Late leavers. Need car charged in the morning sometime. Car not available before 24:00.

Presume users also need to use their cars the next day. Users need a minimum amount of charge. Assume
they can charge some other place during the day so will only top up from home station beyond some minimum
if it is cheap.
'''
from device_kit import *
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__ + '/../')))


meta = {
  'title': "20 EVs with 4 different charge windows\nearly(x3; 0-6am)/mid(x2; 0-12pm)/late(x1; 0-4pm)"
}


def make_deviceset():
  cbounds = (0, 8)
  # All len hardcoded at 24.
  profiles = {
  'early': [
    [2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2.]
  ],
  'mid': [
    [2., 2., 2., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2.]
  ],
  'late': [
    [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
  ]}
  devices = []
  for l, m in {'p0': 0, 'p1': 1}.items():
    for k, i in {'early': 2, 'mid': 3, 'late': 2}.items():
      for j in range(0, i):
        profile = profiles[k][m]
        max_capacity = sum(profile)
        if max_capacity > 0:
          bounds = np.stack((np.zeros(24), profiles[k][m]), axis=1)
          cbounds = [min(3, max_capacity), max_capacity]
          _type = np.random.randint(0, 4)
          params = {'p_h': -_type/4, 'p_l': -1}
          devices.append(CDevice2("%s-%s-%02d-%d" % (l, k, j, _type), 24, bounds, cbounds, **params))
  devices.append(make_gen())
  return DeviceSet('site', devices, sbounds=(0, 100))


def make_gen():
  ''' '''
  return GDevice(
    'supply',
    24,
    np.stack((-100 * np.ones(24), np.zeros(24)), axis=1),
    **{'cost_coeffs': np.array([0.00045, 0.0058, 0.024, 0])*10}
  )


# def matplot_network_writer_hook(event, plt, writer):
#   if event == 'after-init':
#     writer.ymax = 20
#     writer.ymin = -20
