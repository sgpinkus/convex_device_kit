import numpy as np
from device_kit import *
import matplotlib.pyplot as plt

basis = 10

d1 = IDevice2(
  'd1',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'p_l': -1,
    'p_h': 0,
  }
)
d2 = IDevice2(
  'd2',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'p_l': -2,
    'p_h': 0,
  }
)
d3 = IDevice2(
  'd3',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'p_l': -1,
    'p_h': -0.5,
  }
)
d4 = IDevice2(
  'd4',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'p_l': -2,
    'p_h': -0.5,
  }
)


def cost(x, d):
  return d.costv(np.ones(basis)*x, 0)[0]


def deriv(x, d):
  return d.deriv(np.ones(basis)*x, 0)[0]


x = np.linspace(-0.01, 1.01, 100)
plt.plot(x, [cost(v, d1) for v in x], label='d1', color='red')
plt.plot(x, [cost(v, d2) for v in x], label='d2', color='green')
plt.plot(x, [cost(v, d3) for v in x], label='d3', color='blue')
plt.plot(x, [cost(v, d4) for v in x], label='d4', color='purple')
plt.plot(x, [deriv(v, d1) for v in x], label='dd1', color='red', linestyle='--')
plt.plot(x, [deriv(v, d2) for v in x], label='dd2', color='green', linestyle='--')
plt.plot(x, [deriv(v, d3) for v in x], label='dd3', color='blue', linestyle='--')
plt.plot(x, [deriv(v, d4) for v in x], label='dd4', color='purple', linestyle='--')

plt.legend(
  prop={'size': 12},
  # loc='upper right',
  framealpha=0.6,
  frameon=True,
  fancybox=True,
  borderaxespad=-3
)

plt.show()
