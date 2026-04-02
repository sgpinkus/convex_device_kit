import numpy as np

from .basedevice import BaseDevice
from .mfdeviceset import MFDeviceSet


class TwoRatioMFDeviceSet(MFDeviceSet):
  ''' Add a constraint saying flow one must equal k times flow two. Only 2 flows supported because
  adding the constraint for >2 flow is more difficult and I don't need it.
  '''
  ratios = None
  constraint_type = 'eq'

  def __init__(self, device: BaseDevice, flows, ratios, constraint_type='eq'):
    super().__init__(device, flows)
    if len(flows) != 2:
      raise ValueError('More than two flows not supported.')
    if ratios is not None and not len(ratios) == len(flows):
      raise ValueError('Flows and flow ratios must have same length')
    if constraint_type not in ['eq', 'ineq']:
      raise ValueError('Invalid constraint type')
    self.ratios = ratios
    self.constraint_type = constraint_type

  def slice(self, history):
    '''Slice the wrapped device and rebuild with same flows, ratios and constraint_type.'''
    history = np.asarray(history)
    T = history.shape[1]
    if T >= len(self):
      raise ValueError(f'History length {T} must be less than device length {len(self)}')
    combined_history = history.sum(axis=0, keepdims=True)
    sliced_device = self._device.slice(combined_history)
    return TwoRatioMFDeviceSet(sliced_device, self._flows, self.ratios, self.constraint_type)

  @property
  def constraints(self):
    constraints = super().constraints
    shape = self.shape
    flat_shape = shape[0]*shape[1]
    for i in range(0, len(self)):  # for each time
      constraints += [{
        'type': self.constraint_type,
        'fun': lambda s, i=i, r=self.ratios: s.reshape(shape)[0, i]*r[0] - s.reshape(shape)[1, i]*r[1],
        'jac': lambda s, i=i, r=self.ratios: zmm(s.reshape(shape), i, axis=1, fn=lambda x: np.array([r[0], -r[1]])).reshape(flat_shape)
      }]
    return constraints

  def to_dict(self):
    ''' Dump object as a dict. '''
    d = super().to_dict()
    d.update({
      'ratios': self.ratios,
      'constraint_type': self.constraint_type
    })
    return d
