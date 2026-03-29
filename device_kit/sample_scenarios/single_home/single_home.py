'''
Single home with AC, EV, Dimmable lighting, Pool pump, Battery, PV, and unschedulable load. The
Unschedulable load is presumed to be that of a number of devices.

This scenario is reentrant.
'''
import numpy as np
from numpy import array, ones, zeros, stack, hstack, vstack, linspace, sin, pi
from device_kit import *


meta = {
  'title': "Single home [7 devs inc storage and PV], single supply [unconstrained quadratic]"
}


def make_deviceset():
  return DeviceSet('nw', [
    DeviceSet(
      'home',
      make_home_devices(),
      sbounds=(0, 100),
    ),
    make_supply_device()
  ])


def make_home_devices():
  phev_plugged = hstack((zeros(10), ones(14)))  # {18-24,1-7}
  need_lights = hstack((zeros(10), ones(5), zeros(9)))  # {18-23}
  unschedulable_loads = array([+102.817, +155.100, +235.217, +260.783, +168.850, +286.700, +201.950, +278.683, +323.950, +181.633, +102.317, +103.350, +102.750, +102.267, +102.183, +101.767, +101.267, +104.550, +105.083, +101.800, +141.400, +100.883, +100.517, +123.800])/1000
  pv_max_rate = 2
  pv_area = 3
  pv_efficiency = 0.9
  pv_solar_intensity = np.maximum(0, np.sin(linspace(0, np.pi*2, 24)))
  return [
    TDevice(
      'aircon',
      24,
      stack((zeros(24), 4*ones(24)), axis=1),
      **{
        't_external': [25, 27, 30, 33, 34, 36, 33, 32, 31, 30, 29, 28, 28, 27, 26, 24, 22, 22, 21, 21, 22, 22, 23, 25],
        't_init': 25,
        't_optimal': 23,
        't_range': 3,
        'sustainment': 0.80,
        'efficiency': -8
      }
    ),
    CDevice(
      'ecar',
      24,
      stack((zeros(24), 2*phev_plugged), axis=1),
      [5, 10],
      **{'a': -0.25, 'b': 0}
    ),
    CDevice(
      'pump',
      24,
      stack((zeros(24), 1.5*ones(24)), axis=1),
      [1.4, 2.5],
      **{'a': -0.25, 'b': 0}
    ),
    IDevice(
      'lighting',
      24,
      stack((need_lights*0.2, need_lights*0.8), axis=1),
      None,
      **{'a': 0, 'b': 4, 'c': 0.25}
    ),
    Device(
      'unsched',
      24,
      stack((unschedulable_loads, unschedulable_loads), axis=1),
    ),
    SDevice(
      'battery',
      24,
      stack((-2*ones(24), 2*ones(24)), axis=1),
      None,
      **{
        'c1': 0.1,
        'c2': 0.05,
        'c3': 0.001,
        'damage_depth': 0.2,
        'start': 0.5,
        'reserve': 0.5,
        'capacity': 7,
        'efficiency': 0.8,
      }
    ),
    PVDevice(
      'solar',
      24,
      stack((-1*np.minimum(pv_max_rate, pv_solar_intensity*pv_efficiency*pv_area), zeros(24)), axis=1),
    )
  ]


def make_supply_device():
  ''' Cost is $/kW '''
  return GDevice(
    'supply',
    24,
    stack((-50 * ones(24), zeros(24)), axis=1),
    None,
    # {'cost_coeffs': stack((sin(linspace(0,pi,dimension))*0.001+0.005, ones(dimension)*0.005, zeros(dimension)), axis=1)
    **{'cost_coeffs': [0.06, 0.024, 0]}
  )
