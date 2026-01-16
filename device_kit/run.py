#!/usr/bin/env python3
''' Convenience script to just solve for outright cost minimized balanced flow - no market sim crap.
'''
from os import mkdir
from os.path import splitext, basename, exists
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import device_kit
from device_kit.loaders import builder_loader, module_loader
# from device_kit.utils import get_device_by_id
from device_kit.plots import *


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np_printoptions = {
  'linewidth': 1e6,
  'threshold': 1e6,
  'formatter': {
    'float_kind': lambda v: '%+0.3f' % (v,),
    'int_kind': lambda v: '%+0.3f' % (v,),
  },
}
np.set_printoptions(**np_printoptions)
pd.set_option('display.float_format', lambda v: '%+0.3f' % (v,),)


def main():
  parser = argparse.ArgumentParser(
    description='Run a device_kit scenario and print result to stdout',
    prog='device_kit'
  )
  parser.add_argument('filename', action='store',
    help='name of a python module file containing scenario to run'
  )
  parser.add_argument('--loader', '-l', action='store', default='module', type=str,
    help='loader to use to load scenario file')
  args = parser.parse_args()
  loader = globals()[f'{args.loader}_loader']
  (deviceset, meta, cb) = loader.load_file(args.filename)
  deviceset.sbounds = (0,0)
  print(str(deviceset))
  (x, solve_meta) = device_kit.solve(deviceset, solver_options={'ftol': 1e-6}, cb=Cb()) # Convenience convex solver.
  print(solve_meta.message)

  df = pd.DataFrame.from_dict(dict(deviceset.map(x)), orient='index')
  df.loc['excess'] = df.sum()
  print('Flows:')
  print(df.sort_index())
  print('Derivatives:')
  df_derivs = pd.DataFrame.from_dict(dict(deviceset.map(deviceset.deriv(x, 0))), orient='index')
  print(df_derivs.sort_index())

  # x = dict(deviceset.map(x))['nw.supply']
  # d = get_device_by_id(deviceset, 'supply')
  # print(d.costv(x, 0))
  # print(d.deriv(x, 0))

  df.drop('excess', axis=0, inplace=True)

  if not exists('run-out/'):
    mkdir('run-out')
  output_filename = 'run-out/' + splitext(basename(args.filename))[0]

  title = meta.get('title') if meta else None
  fig, ax = plot_dataframe_as_stacked_bars(df, aggregation_level=3)

  ax.plot(range(0, df.shape[1]), df.sum(), label='excess', color='red')
  df.loc['excess'] = df.sum()

  for (label, d) in df_derivs.iterrows():
    if label.find('supply') >= 0:
      ax.plot(d.index, d, label=label + '_deriv')

  _len = df.shape[1]
  ax.set_xlim(-2, _len+2)
  # ax.set_ylim(df.min(), df.max())
  ax.set_title(title)
  ax.legend(
    prop={'size': 10},
    # loc='upper right',
    framealpha=0.6,
    frameon=True,
    fancybox=True,
    borderaxespad=-3
  )

  ax.grid(
      True,
      which='both',      # 'major', 'minor', or 'both'
      linestyle='--',
      linewidth=0.5,
      color='gray',
      alpha=0.7
  )
  fig.savefig(output_filename + '.png', format='png')
  df.to_csv(output_filename + '.csv', float_format='%.3f')

  print('Total Costs')
  print(pd.Series({ _id: d.cost(_x, 0) for (_id, d, _x) in deviceset.mapDevices(x)}))
  print('Total Flows')
  print(df.sum(axis=1))


class Cb():
  def __init__(self):
    self.i = 0

  def __call__(self, device, x):
    logger.info('step=%d; cost=%.6f' % (self.i, device.cost(x, 0)))
    self.i += 1


if __name__ == '__main__':
  main()
