import numpy as np
from device_kit import *
import matplotlib.pyplot as plt

basis = 10

d1 = GDevice(
  'd1',
  basis,
  np.stack((-1*np.ones(basis), np.zeros(basis)), axis=1),
  **{
    'cost_coeffs': [1, 1, 0],
  }
)


def cost(x, d):
  return d.costv(np.ones(basis)*x, 0)[0]


def deriv(x, d):
  return d.deriv(np.ones(basis)*x, 0)[0]


x = np.linspace(-1.01, 0.01, 100)
plt.plot(x, [cost(v, d1) for v in x], label='d1', color='red')
plt.plot(x, [deriv(v, d1) for v in x], label='dd1', color='red', linestyle='--')

plt.legend(
  prop={'size': 12},
  # loc='upper right',
  framealpha=0.6,
  frameon=True,
  fancybox=True,
  borderaxespad=-3
)

plt.show()
