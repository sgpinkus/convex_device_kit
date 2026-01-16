import sys
import os
from random import random, randrange, seed
import numpy as np
from device_kit import *


meta = {
  'title': 'x10 aircons with random preferences in range [20-25][+-][1-4]'
}
dimension = 24

def make_deviceset():
  cost = np.ones((dimension, 4))*[0.045, 0.58, 0.24, 0]
  bounds = (0,4)
  return DeviceSet('site', [
    TDevice('ac01', dimension, bounds, **params()),
    TDevice('ac02', dimension, bounds, **params()),
    TDevice('ac03', dimension, bounds, **params()),
    TDevice('ac04', dimension, bounds, **params()),
    TDevice('ac05', dimension, bounds, **params()),
    TDevice('ac06', dimension, bounds, **params()),
    TDevice('ac07', dimension, bounds, **params()),
    TDevice('ac08', dimension, bounds, **params()),
    TDevice('ac09', dimension, bounds, **params()),
    TDevice('ac10', dimension, bounds, **params()),
    GDevice('supply', dimension, np.stack((-100*np.ones(dimension), np.zeros(dimension)), axis=1), **{'cost_coeffs': cost})
  ],
  sbounds=(0,123)
  )


def params():
  return {
    't_external': np.sin(np.linspace(0, 2*np.pi, dimension))*15+15,
    't_init': 15,
    't_optimal': randrange(20, 25)+random()*0.25,
    'care_time': np.ones(dimension),
    't_range': randrange(1, 4),
    'sustainment': 0.95,
    'efficiency': 0.5+random()
  }


def params_degenerate():
  ''' Outcome will be slightly worse with theses params in e951f6c0. I dont know why exactly. High
   `t_b` and `a` make the error more likely. Also, high `a` gives the dreaded '+ve directional ...'
  on initialization.
  '''
  return {
    't_external': np.sin(np.linspace((1/6.)*np.pi, (7/6.)*np.pi, dimension))*15+70,
    't_init': 78,
    't_optimal': 75,
    't_range': 2,
    'sustainment': 0.99,
    'efficiency': -8,
  }


# def matplot_network_writer_hook(event, fig, writer):
#   if event == 'after-init':
#     writer.ymax = 32
#     writer.ymin = -10
#   if event == 'after-update':
#     colors = writer._colors
#     fig.axes[0].plot(agents[0].t_external, label='temp_external')
#     for i, a in enumerate(agents):
#       if hasattr(a, 't_optimal'):
#         fig.axes[0].axhline(a.t_optimal, color=colors[i%len(colors)], ls='--')
#         fig.axes[0].plot(a.r2t(a.r), label='temp_'+a.id, color=colors[i%len(colors)], lw=1.2)
#     fig.axes[0].ylim(fig.axes[0].ylim()[0], max(fig.axes[0].ylim()[1], agents[0].t_external.max()))
