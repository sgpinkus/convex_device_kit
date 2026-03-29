'''
This scenario represents the scenario tested in the original paper.

  - Households are of 1 of 2 types. Type defines the usage profile characteristics.
    - Type 0: Get home at 18:00 and start using / caring about things.
    - Type 1: Stay at home all day.
  - 8 houses 4 of each type.
  - Day is 24H and starts at 8am for some reason.
  - Each house has 5 appliances (plus a battery):
    - AC, PHEV, Clothes Washer, Lighting, Entertainment.
  - Unit of power is kW *not* Watts.

Note subsequently added type 3; like type 1 but greedy and uncompromising.
'''
from random import seed, random, randrange
import numpy as np
from numpy import array, zeros, ones, stack, hstack, concatenate
from device_kit import *


seed(23)
meta = {
  'title': 'The LCL reference scenario; 8 houses split into 2 types; \nSame appliance set of 6 appliances; randomized settings within bounds'
}


def irandom():
  ''' Random fraction N/16. Same diff and makes print outs easier to read. '''
  return randrange(0, 17)*(1/16.)


def make_deviceset():
  return DeviceSet('mg',
                   [
      make_home(type=1, id='home1'),
      make_home(type=1, id='home2'),
      make_home(type=1, id='home3'),
      make_home(type=1, id='home4'),
      make_home(type=2, id='home5'),
      make_home(type=2, id='home6'),
      make_home(type=2, id='home7'),
      make_home(type=2, id='home8'),
      make_gas_gen(type=1, id='supply')
    ]
  )


def make_home(type, id, battery=True, pv=False, sbounds=(0, 100)):
  devices = [
    make_ac(type, id),
    make_phev(type, id),
    make_washer(type, id),
    make_lighting(type, id),
    make_entertainment(type, id),
  ]
  if battery:
    devices += [make_battery(type, id)]
  if pv:
    devices += [make_pv(type, id)]
  return DeviceSet(id, devices, sbounds=sbounds)


def make_ac(type, id):
  ''' Temperature range shown is 'typical' day in Cali. The temp shown has range of [71,98] degrees.
  Note there is two ways to interpret care time (`care`), which is
  supposed to model when the user is home and thus cares about the temperature:

    1. The utility of temperature is zero at those times.
    2. The max power consumption of the ac is zero at those times.

  Alternate 1 leads to unrealistic drastic attempt to precool in don't care about price scenario.
  Clearly LCL did not use this interpretation beacuse this drastic precooling is not shown in their
  result plots. So we are going with alternate 2.
  We also use KW not Watts.
  '''
  p_range = [0, 4]
  care = ones(24)
  params = {
    't_external': array([80, 84, 89, 94, 96, 100, 95, 92, 90, 89, 87, 86, 85, 83, 81, 78, 75, 74, 73, 73, 74, 75, 77, 79])-2,
    't_init': 78,
    't_optimal': randrange(73, 77),  # randrange(73,78),
    't_range': randrange(1, 4),
    'sustainment': 1-0.90,
    'efficiency': -8-3*irandom(),  # 1000*[−0.011, −0.008]
    'c': care*0.025
  }
  bounds = stack((p_range[0]*care, p_range[1]*care), axis=1)
  if type == 2:
    care = hstack((zeros(10), ones(14)))  # {18-24,1-7}
    bounds = stack((p_range[0]*care, p_range[1]*care), axis=1)
  elif type == 3:
    params['t_optimal'] = 73
    params['t_range'] = 0.5
    params['c'] = care*0.25
  return TDevice('aircon', 24, bounds, **params)


def make_phev(type, id):
  '''  No diff between type for PHEV. '''
  care = hstack((zeros(10), ones(14)))  # {18-24,1-7}
  p_range = [0, 2]
  bounds = stack((p_range[0]*care, p_range[1]*care), axis=1)
  cbounds = [4.8+irandom()*0.3, 5.5+irandom()*0.5]
  params = {'a': -0.25, 'b': 0}
  if type == 3:
    params = {'a': -2.5, 'b': 0}
  return CDevice('ecar', 24, bounds, cbounds, **params)


def make_washer(type, id):
  p_range = [0, 1.5]
  care = ones(24)
  cbounds = [1.4+irandom()*0.2, 2+irandom()*0.5]
  params = {'a': -0.25, 'b': 0}
  bounds = stack((p_range[0]*care, p_range[1]*care), axis=1)
  if type == 2:
    care = hstack((zeros(10), ones(14)))  # {18-24,1-7}
    bounds = stack((p_range[0]*care, p_range[1]*care), axis=1)
  elif type == 3:
    params = {'a': -2.5, 'b': 0}
  return CDevice('washer', 24, bounds, cbounds, **params)


def make_lighting(type, id):
  ''' No diff betweeen type 1, 2 for lighting. '''
  care = hstack((zeros(10), ones(5), zeros(9)))  # {18-23}
  p_range = [0.2, 0.8]
  bounds = stack((care*p_range[0], care*p_range[1]), axis=1)
  params = {'a': 0, 'b': 4, 'c': 0.25}
  if type == 3:
    params = {'a': care*0.1, 'b': 4, 'c': 2.5}
  return IDevice('lighting', 24, bounds, **params)


def make_entertainment(type, id):
  p_range = [0, 0.4]
  params = {'a': 0, 'b': 4, 'c': 0.25}
  care = hstack((zeros(4), ones(11), zeros(9)))  # {12-23}
  bounds = stack((p_range[0]*care, p_range[1]*care), axis=1)
  cbounds = [1.2, 3.5]
  if type == 2:
    care = hstack((zeros(10), ones(6), zeros(8)))  # {18-24}
    bounds = stack((p_range[0]*care, p_range[1]*care), axis=1)
    cbounds = [0.5, 2.0]
  elif type == 3:
    params = {'a': care*0.1, 'b': 4, 'c': 2.5}
  return IDevice('tv-av', 24, bounds, cbounds, **params)


def make_battery(type, id, max_rate=1.8, capacity=5.5):
  ''' No diff for battery '''
  params = {'c1': 0.1, 'c2': 0.0, 'c3': 0.0, 'damage_depth': 0.1, 'reserve': 0.5, 'capacity': capacity+irandom()}
  return SDevice('battery', 24, [-max_rate, max_rate], **params)


def make_pv(type, id, max_rate=3, area=5, efficiency=0.9):
  ''' Not part of LCL scenario. '''
  solar_intensity = np.maximum(0, np.sin(np.linspace(0, np.pi*2, 24)))
  lbounds = -1*np.minimum(max_rate, solar_intensity*efficiency*area)
  return PVDevice('solar', 24, (lbounds, np.zeros(24)))


def make_gas_gen(type, id):
  ''' The single supply in the Li, Chen Low reference paper is some kind of thermal generator with a
  piecewise quadratic. Any continuous and c.d. piecewise quadratic can be approximated witha higer
  order polynomial. Li Na provided the cost function they used. The 3-nomial below was fitted to
  that. Note cost functionis kW->$.
  '''
  cost = [0.00045, 0.0058, 0.024, 0]
  device = GDevice(
    id,
    24,
    stack((-50 * ones(24), zeros(24)), axis=1),
    **{'cost_coeffs': cost}
  )
  return device
