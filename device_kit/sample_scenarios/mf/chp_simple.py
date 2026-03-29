'''
Multi flow CHP scenario. BROKEN.
'''
import numpy as np
from collections import OrderedDict
from device_kit import *


def make_deviceset():
  cost = np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)
  devices = OrderedDict([
      ('uncntrld', Device('uncntrld', 24, (random_uncntrld(),))),
      ('scalable', IDevice2('scalable', 24, (0, 2), (12, 18))),
      ('shiftable', CDevice('shiftable', 24, (0, 2), (12, 24))),
      ('generator', GDevice('generator', 24, (-50, 0), None, **{'cost_coeffs': cost})),
  ])
  return SubBalancedDeviceSet('site1', [
      # devices['uncntrld'],
      MFDeviceSet(devices['scalable'], ['e', 'h']),
      devices['shiftable'],
      TwoRatioMFDeviceSet(devices['generator'], ['e', 'h'], [1, 8]),
      Device('slack.h', 24, (-100, 100)),
    ],
    labels=['h'],
    constraint_type='eq'
  )


def random_uncntrld():
  np.random.seed(109)
  return np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1, 1, 24)))
