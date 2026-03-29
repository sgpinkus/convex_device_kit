import re
import numpy as np
from collections import OrderedDict
from device_kit import *


class SubBalancedDeviceSet(DeviceSet):
  ''' Find all devices under this device set matching ".*{label}$" and apply an additional balancing
  constraint. I.e. so labelled devices flows sum to 0.
  TODO: Add sbounds option for sub flows. The sub flows are hard coded to sum to 0. Currently sbounds
  belongs to the parent DeviceSet and specifies constraint over all flows regardless of types (label).
  '''
  def __init__(self, id, devices, sbounds=None, labels=[], constraint_type='eq', sign=1, apply_to_remaining=False):
    super().__init__(id, devices, sbounds)
    self.labels = labels
    self.constraint_type = constraint_type
    self.sign = sign
    self.apply_to_remaining = apply_to_remaining
    (self.labelled_sets, self.unlabelled_set) = self._labelled_sets()
    self.labelled_sets = list(self.labelled_sets.values())

  @property
  def constraints(self):
    constraints = super().constraints
    shape = self.shape
    sets = self.labelled_sets + ([self.unlabelled_set] if self.apply_to_remaining else [])
    for labelled_set in sets:
      col_jac = np.zeros(shape[0])
      col_jac[labelled_set] = 1
      for i in range(0, len(self)):  # for each time
        constraints += [{
          'type': self.constraint_type,
          'fun': lambda s, i=i: self.sign*(s.reshape(shape)[:, i]*col_jac).sum(),
          # 'jac': lambda s, i=i, j=col_jac: self.sign*zmm(s.reshape(shape), i, axis=1, fn=lambda r: j).reshape(flat_shape)
        }]
    return constraints

  def _labelled_sets(self):
    ''' Return hash, (label, indexes) where indexes is the rows of this device with that label. '''
    leaf_devices = OrderedDict(self.leaf_devices())
    labelled = {}
    unlabelled = set(range(len(leaf_devices)))
    for label in self.labels:
      labelled[label] = [k for k, v in enumerate(leaf_devices.keys()) if re.match('.*{label}$'.format(label=label), v)]
      unlabelled.difference_update(labelled[label])
    return (labelled, list(unlabelled))

  def to_dict(self):
    ''' Dump object as a dict. '''
    d = super().to_dict()
    d.update({
      'labels': self.labels,
      'constraint_type': self.constraint_type,
      'sign': self.sign,
      'apply_to_remaining': self.apply_to_remaining,
    })
    return d
