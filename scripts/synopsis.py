import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from device_kit import *


def random_uncontrolled():
  return np.maximum(0, 0.5+np.cumsum(np.random.uniform(-1, 1, 24)))


def generator_cost_curve():
  return np.stack((np.sin(np.linspace(0, np.pi, 24))*0.5+0.1, np.ones(24)*0.001, np.zeros(24)), axis=1)


def make_model():
  ''' Small power network model. '''
  np.random.seed(19)
  pv_max_rate = 2
  pv_area = 3
  pv_efficiency = 0.9
  pv_solar_intensity = np.maximum(0, np.sin(np.linspace(0, np.pi*2, 24)))
  model = DeviceSet('site1', [
      Device('uncontrolled', 24, (random_uncontrolled(),)),
      IDevice2('scalable', 24, (0.25, 2), (0, 24), p_h=-0.5),
      CDevice('shiftable', 24, (0, 2), (12, 24), a=-1.),
      GDevice('generator', 24, (-10, 0), cbounds=None, cost_coeffs=generator_cost_curve()),
      PVDevice(
        'solar',
        24,
        np.stack((-1*np.minimum(pv_max_rate, pv_solar_intensity*pv_efficiency*pv_area), np.zeros(24)), axis=1),
      ),
      SDevice('battery', 24, (-5, 5), c1=0.01, capacity=14, sustainment=1.0, efficiency=0.99)
    ],
    sbounds=(0, 0)  # balanced flow constraint.
  )
  return model


def main():
  model = make_model()
  (x, solve_meta) = solve(model, p=0)  # Convenience convex solver.
  print(solve_meta.message)
  df = pd.DataFrame.from_dict(dict(model.map(x)), orient='index')
  df.loc['total'] = df.sum()
  supply_side, _slice = [(d, s) for (d, s) in model.slices if d.id == 'generator'][0]
  df.loc['price'] = -1*supply_side.deriv(x[slice(*_slice), :], p=0)
  pd.set_option('display.float_format', lambda v: '%+0.3f' % (v,),)
  print(df.sort_index())
  print('Cost: ', model.cost(x, p=0))
  df.transpose().plot(drawstyle='steps', grid=True)
  plt.ylabel('Power (kWh)')
  plt.xlabel('Time (H)')
  plt.savefig('synopsis.png')


if __name__ == '__main__':
  main()
