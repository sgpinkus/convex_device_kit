''' A GUI builder exist that makes devices with this format. '''
import json
from os.path import basename
import importlib
import numpy as np
import argparse
import device_kit
from device_kit.functions import *
from device_kit.utils import flatten
import logging


logger = logging.getLogger()


def main():
  _self = importlib.import_module('__main__')
  parser = argparse.ArgumentParser(
    description='Load a device_kit scenario from some ambiguous builder export',
  )
  parser.add_argument('filename', action='store',
    help='name of JSON file containing scenario to load'
  )
  args = parser.parse_args()
  logger.info(load_file(args.filename))


def load_file(filename):
  data = json.load(open(filename, 'r'))
  return (load_data(data), {}, None)


def load_data(data):
  basis = data['basis']
  data = data['devices']
  devices = []
  for d in data:
    loader = globals()['load_%s_device' % (d['type'],)]
    logger.info('found device with type=%s' % (d['type'],))
    devices.append(loader(d, basis))
  return device_kit.DeviceSet(data['name'] if 'name' in data else 'site', devices)


def load_load_device(d, basis: int):
  device_id = d['title'] if 'title' in d else d['type']
  bounds = run_to_array(d['bounds'])
  cbounds = load_cbounds(d)
  params = {}
  cost_function = load_cost_function(d, bounds, cbounds, basis)
  if cost_function:
    params = { 'f': cost_function }
  return device_kit.ADevice(device_id, basis, bounds, cbounds, **params)


def load_fixed_load_device(d, basis: int):
  device_id = d['title'] if 'title' in d else d['type']
  #, id={device_id}')
  bounds = run_to_array(d['bounds'])
  if (bounds[:,0] != bounds[:,1]).all():
    raise Exception('Invalid fixed load device')
  return device_kit.ADevice(device_id, basis, bounds)


def load_storage_device(d, basis: int):
  parameter_map = {
    'capacity': 'capacity',
    'efficiencyFactor': 'efficiency',
    'reserveRatio': 'reserve',
    'startingRatio': 'start',
    'fastChargeCostFactor': 'c1',
    'flipFlopCostFactor': 'c2',
    'deepDischargeCostFactor': 'c3',
    'deepDepthRatio': 'damage_depth',
    # chargeRateClippingFactor?: number,
    # disChargeRateClippingFactor?: number,
  }
  device_id = d['title'] if 'title' in d else d['type']
  bounds = run_to_array(d['bounds'])
  params = { parameter_map[k]: v for k, v in d['parameters'].items() }
  rate_clip = (None, None)
  if 'disChargeRateClippingFactor' in d['parameters']:
    rate_clip[0] = d['parameters']['disChargeRateClippingFactor']
  if 'chargeRateClippingFactor' in d['parameters']:
    rate_clip[1] = d['parameters']['chargeRateClippingFactor']
  params['rate_clip'] = rate_clip
  return device_kit.SDevice(device_id, basis, bounds, **params)


def load_supply_device(d, basis: int):
  device_id = d['title'] if 'title' in d else d['type']
  bounds = -1*run_to_array(d['bounds'])
  bounds = np.array([bounds[:,1], bounds[:,0]])
  cbounds = load_cbounds(d)
  params = {}
  cost_function = load_cost_function(d, bounds, cbounds, basis)
  if cost_function:
    params = { 'f': ReflectedFunction(cost_function) }
  return device_kit.ADevice(device_id, basis, bounds, cbounds, **params)

def load_thermal_load_device(d, basis: int):
  parameter_map = {
    'desiredTemperature': 't_optimal',
    'initialTemperature': 't_init',
    'thermalSustainment': 'sustainment',
    'efficiencyFactor': 'efficiency',
    'externalTemperatureProfile': 't_external',
    'temperatureVariationCareFactor': 't_range'
  }
  device_id = d['title'] if 'title' in d else d['type']
  bounds = run_to_array(d['bounds'])
  params = { parameter_map[k]: v for k, v in d['parameters'].items() }
  params['t_external'] = run_to_array(d['parameters']['externalTemperatureProfile'])
  params['t_range'] = run_to_array(d['parameters']['temperatureVariationCareFactor'])
  return device_kit.TDevice(device_id, basis, bounds, **params)


def load_cbounds(d):
  if 'cumulative_bounds' in d:
    logger.info(f'\tfound cbounds: {d['cumulative_bounds']}')
    return run_to_cbounds_array(d['cumulative_bounds'])


def load_cost_function(d, bounds, cbounds, basis):
  costs_data = d['costs']
  costs = []
  if 'flow' in costs_data:
    logger.info('\tfound flow cost for %s' % (d['type'],))
    coeffs = _reshape_offset_quad_coeffs(run_to_array(costs_data['flow']))
    costs += [Poly2DOffset(coeffs)]
  if 'cumulative_flow' in costs_data:
    logger.info('\tfound cumulative_flow for %s' % (d['type'],))
    raise Exception('Not implemented: costs.cumulative_flow')
  if 'flow_bounds_relative' in costs_data:
    logger.info('\tfound flow_bounds_relative cost for %s' % (d['type'],))
    _functions = [HLQuadraticCost(v[0], v[1], bounds[i, 0], bounds[i, 1]) for i, v in enumerate(run_to_array(costs_data['flow_bounds_relative']))]
    costs += [X2D(_functions)]
  if 'cumulative_flow_bounds_relative' in costs_data:
    logger.info('\tfound cumulative_flow_bounds_relative cost for %s: %s', (d['type'], costs_data['cumulative_flow_bounds_relative']))
    p_l, p_h = costs_data['cumulative_flow_bounds_relative']
    costs += [RangesFunction([((c[2], c[3]), InnerSumFunction(HLQuadraticCost(p_l, p_h, c[0], c[1]))) for c in cbounds])]
  if 'peak_flow' in costs_data:
    logger.info('\tfound peak_flow cost for %s' % (d['type'],))
    costs += [DemandFunction(np.poly1d(costs_data['peak_flow']))]
  return SumFunction(costs) if len(costs) else None


def _reshape_offset_quad_coeffs(x):
  x = np.array(x)
  basis = len(x)
  return np.concatenate((x[:, 0:2], np.zeros((basis, 1)), np.array(x[:, 2]).reshape((basis, 1))), axis=1)


def run_to_array(run):
  item_template = run['runs']['0']
  shape = (run['basis'], len(item_template)) if hasattr(item_template, '__len__') else (run['basis'],)
  _array = np.zeros(shape)
  points = list(run['runs'].keys())
  for i,v in enumerate(points):
    e = int(points[i+1]) if i < len(points) - 1 else run['basis']
    _array[int(v):e] = run['runs'][v]
  return _array


def run_to_cbounds_array(run):
  _array = []
  points = list(run['runs'].keys())
  for i,v in enumerate(points):
    [l, h] = run['runs'][v]
    e = int(points[i+1]) if i < len(points) - 1 else run['basis']
    _array.append([l,h,int(v),e])
  return _array


if __name__ == '__main__':
  main()