'''
Case study LCL scenario with some changes:

  - Export is allowed.
  - Every house has PV.
  - Every 2nd house has a battery (PV is more common than batteries).
  - The MG owner / "utility company" has a large PV and battery in addition to usual supply.
'''
import numpy as np
from device_kit import *
from device_kit.sample_scenarios.lcl.lcl_scenario import *


meta = {
  'title': 'The LCL reference scenario with prosumption'
}


def make_deviceset():
  return DeviceSet('mg', [
      make_home(type=1, id='home1', sbounds=(-10, 10)),
      make_home(type=1, id='home2', pv=True, sbounds=(-10, 10)),
      make_home(type=1, id='home3', sbounds=(-10, 10)),
      make_home(type=1, id='home4', pv=True, sbounds=(-10, 10)),
      make_home(type=2, id='home5', sbounds=(-10, 10)),
      make_home(type=2, id='home6', pv=True, sbounds=(-10, 10)),
      make_home(type=2, id='home7', sbounds=(-10, 10)),
      make_home(type=2, id='home8', pv=True, sbounds=(-10, 10)),
      make_supply()
    ]
  )


def make_supply(type=1, id='supply'):
  ''' This doesn't work with LCL mechanism '''
  return DeviceSet(id,
                   devices=[
      make_gas_gen(type, 'gas'),
      make_battery(type, 'battery', max_rate=10, capacity=20),
      make_pv(type, 'pv', max_rate=10, area=20, efficiency=0.9)
    ],
    sbounds=(-100, 100)
  )


def make_battery(type, id, max_rate=1.8, capacity=5.5):
  ''' No diff for battery '''
  p_range = [-max_rate, max_rate]
  bounds = stack((np.ones(24)*p_range[0], np.ones(24)*p_range[1]), axis=1)
  params = {'c1': 0.1, 'c2': 0.0, 'c3': 0.0, 'damage_depth': 0.1, 'reserve': 0.5, 'capacity': capacity+irandom()}
  return SDevice('battery', 24, bounds, **params)


# def matplot_network_writer_hook(event, plt, writer=None):
#   if event == 'after-update':
#     plt.title('')
#     plt.xlabel('Time (H)')
#     plt.ylabel('Power or Cost (kW or $)')
#   elif event == 'after-init':
#     writer.ymax = 15
#     writer.ymin = -15
