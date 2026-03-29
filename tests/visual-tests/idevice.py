import numpy as np
from device_kit import *
import matplotlib.pyplot as plt

basis = 10

d1 = IDevice(
  'd1',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'b': 1,
  }
)
d2 = IDevice(
  'd2',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'b': 2,
  }
)
d3 = IDevice(
  'd3',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'b': 3,
  }
)
d4 = IDevice(
  'd4',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'b': 4,
  }
)
d5 = IDevice(
  'd5',
  basis,
  np.stack((np.zeros(basis), np.ones(basis)), axis=1),
  **{
    'b': 4,
    'a': 0.4,
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
plt.plot(x, [cost(v, d5) for v in x], label='d5', color='orange')
plt.plot(x, [deriv(v, d1) for v in x], label='dd1', color='red', linestyle='--')
plt.plot(x, [deriv(v, d2) for v in x], label='dd2', color='green', linestyle='--')
plt.plot(x, [deriv(v, d3) for v in x], label='dd3', color='blue', linestyle='--')
plt.plot(x, [deriv(v, d4) for v in x], label='dd4', color='purple', linestyle='--')
plt.plot(x, [deriv(v, d5) for v in x], label='dd5', color='orange', linestyle='--')

plt.legend(
  prop={'size': 12},
  loc='upper right',
  framealpha=0.6,
  frameon=True,
  fancybox=True,
  borderaxespad=-3
)

plt.show()
