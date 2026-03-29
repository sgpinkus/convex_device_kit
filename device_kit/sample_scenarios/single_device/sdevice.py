'''
A battery won't do anything unless there is supply and demand to work with. Need at least one supply
and and demand agent. So this scenario adds a battery (SDevice) to existing single idevice.
See idevice.py
'''
import numpy as np
import logging
from device_kit import *
from device_kit.sample_scenarios.single_device import idevice


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
meta = {
  'title': 'Single IDevice scenario augmented with storage to arbitrage'
}


def make_deviceset():
  devices = list(idevice.make_devices().values())
  battery = make_devices()['battery']
  devices.append(battery)
  return DeviceSet('nw', devices)


def make_devices(dimension=24):
  return {
     # Default parameters.
      'defaults': SDevice(
      'defaults',
      dimension,
      np.stack((np.ones(dimension)*-5, np.ones(dimension)*5), axis=1),
    ),
      # Battery should avoid deep discharge but charges as normal.
      'leadacid': SDevice(
      'leadacid',
      dimension,
      np.stack((np.ones(dimension)*-5, np.ones(dimension)*5), axis=1),
      **{'c3': 1, 'damage_depth': 0.5}
    ),
      # c2 is supposed to have a correcting effect on c1 as well as penalize switching. All up,
      # should see battery charge slightly more than default.
      'high-c2': SDevice(
      'high-c2',
      dimension,
      np.stack((np.ones(dimension)*-5, np.ones(dimension)*5), axis=1),
      **{'c1': 1, 'c2': 0.99}
    ),
      # Fast c/d rate less effect.
      'durable': SDevice(
      'durable',
      dimension,
      np.stack((np.ones(dimension)*-5, np.ones(dimension)*5), axis=1),
      **{'c1': 0.5, 'c2': 0}
    ),
      # Just the quadratic RoC cost.
      'battery': SDevice(
      'battery',
      dimension,
      np.stack((np.ones(dimension)*-5, np.ones(dimension)*5), axis=1),
      **{'c1': 0.5, 'c2': 0, 'c3': 0, 'damage_depth': 0}
    ),
  }
