import numpy as np

from .basedevice import BaseDevice
from .device import Device
from .deviceset import DeviceSet


class MFDeviceSet(DeviceSet):
  ''' Provides an adaptor over some *atomic* device to represent multiple flows to/from the single device.
  The flows themselves become the devices of this device set. The flows, represent different
  conduits servicing the device. By default the device is assumed not to have a
  preference (perfect subs). An example, room heating from electricity or from heat a pipe.
  '''
  _id = None
  _length = None          # Number of time units (not number of devices).
  _sbounds = None         # min/max aggregate power consumption by this agent at ant timeslot.
  _devices = None         # Sub devices of this agent.
  _flows = None           # A sub device is created for each flow.

  def __init__(self, device: BaseDevice, flows):
    ''' Setting reasonable bounds for the flow devices is somewhat tricky. We could use (-inf, inf),
    but that can lead to strangnesses. Instead, MFDeviceSet can only work with a device with all non
    -ve or all non +ve device (i.e. not a two way device) and the flow device bounds are set to
    (lower|0, 0|upper) depending on whether the device is -ve|+ve.
    '''
    if not len(flows):
      raise ValueError('Flows must have at least one element')
    if (device.lbounds < 0).any() and (device.hbounds > 0).any():
      raise ValueError('Device not supported. Device must have strict non -ve/+ve bounds set')
    self._device = device
    self._id = device.id
    self._length = len(device)
    self._sbounds = device.bounds
    self._flows = flows
    self._devices = []
    if (device.lbounds < 0).any():
      bounds = (device.lbounds, np.zeros(len(device)))
    else:  # (device.hbound > 0).any():
      bounds = (np.zeros(len(device)), device.hbounds)
    for flow in flows:
      self._devices.append(Device(flow, len(device), bounds))

  def slice(self, history):
    '''Slice the wrapped device and rebuild MFDeviceSet with the same flow names.
    history shape: (n_flows, T). The combined flow (sum across flows) is used
    to condition the underlying device.'''
    history = np.asarray(history)
    T = history.shape[1]
    if T >= len(self):
      raise ValueError(f'History length {T} must be less than device length {len(self)}')
    combined_history = history.sum(axis=0, keepdims=True)
    sliced_device = self._device.slice(combined_history)
    return self.__class__(sliced_device, self._flows)

  def __len__(self):
    return self._length

  def __str__(self):
    return str(self._device) + str(self._flows)

  def cost(self, s, p):
    s = s.reshape(self.shape)
    return self._device.cost(s.sum(axis=0), 0) + (s*p).sum()

  def deriv(self, s, p):
    s = s.reshape(self.shape)
    return np.repeat(self._device.deriv(s.sum(axis=0), 0).reshape(1, len(self)), self.shape[0], axis=0).reshape(self.shape) + p

  def hess(self, s, p=0):
    return self._device.hess(s.sum(axis=0), 0)

  @property
  def id(self):
    return self._id

  @property
  def length(self):
    return self._length

  @property
  def constraints(self):
    ''' Get scipy.optimize.minimize style constraint list for this aggregate device. The constraints
    are to be applied to a matrix, but each device constraints only apply to a row of that matrix.
    Thus to reuse device constraints need to slice out correct row for input and zero pad other rows.
    '''
    constraints = super().constraints
    device_constraints = self._device.constraints
    shape = self.shape
    flat_shape = shape[0]*shape[1]
    for constraint in device_constraints:
      constraint['fun'] = lambda s, f=constraint['fun']: f(s.reshape(shape).sum(axis=0))
      if 'jac' in constraint:
        constraint['jac'] = lambda s, f=constraint['jac']: np.repeat(f(s.reshape(shape).sum(axis=0)), shape[0], axis=0).reshape(flat_shape)
      constraints += [constraint]
    return constraints

  def project(self, s):
    ''' The restrictions on the flowed device's bounds (all non -ve|+ve. See __init__) make it a bit
    easier to come up with a projection satisfying at least bounds and cbounds. Wlog consider a non
    -ve device, given (0,ub) is feasible for all flow devices, we can proportionally scale onto
    whatever projection the flowed device returns.
    '''
    s = self._device.project(s.reshape(self.shape).sum(axis=0))
    return np.repeat(s/self.shape[0], self.shape[0], axis=0).reshape(self.shape)

  def to_dict(self):
    ''' Dump object as a dict. '''
    return {
      'flows': self._flows,
      'device': self._device,
    }

  def __getattr__(self, name):
    ''' Delegate methods and props to device '''
    return getattr(self._device, name)
