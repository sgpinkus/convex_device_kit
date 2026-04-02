import unittest

import numpy as np

from device_kit import (
    ADevice,
    CDevice,
    Device,
    DeviceSet,
    GDevice,
    IDevice,
    IDevice2,
    SDevice,
    TDevice,
    solve,
)
from device_kit.subbalanceddeviceset import SubBalancedDeviceSet
from device_kit.windowdevice import WindowDevice

N = 24
T = 8
RNG = np.random.default_rng(42)


def _history(x, T=T, noise=0.0):
  h = x[:, :T].copy()
  if noise:
    h += RNG.uniform(-noise, noise, h.shape)
  return h


def _check_slice(device, history):
  sliced = device.slice(history)
  assert len(sliced) == N - T
  assert sliced.shape[0] == device.shape[0]
  assert sliced.bounds.shape == (device.shape[0] * (N - T), 2)
  return sliced


class TestDeviceSlice(unittest.TestCase):
  def test_basic_length_and_bounds(self):
    d = Device('d', N, (0.5, 2.0))
    history = np.full((1, T), 1.0)
    sliced = _check_slice(d, history)
    np.testing.assert_array_equal(sliced.bounds, d.bounds[T:])

  def test_per_slot_bounds_preserved(self):
    bounds = np.stack([np.linspace(0, 0.5, N), np.linspace(1, 2, N)], axis=1)
    d = Device('d', N, bounds)
    history = np.full((1, T), 0.3)
    sliced = d.slice(history)
    np.testing.assert_array_equal(sliced.bounds, bounds[T:])

  def test_cbounds_partially_consumed(self):
    d = Device('d', N, (0.0, 2.0), cbounds=(5.0, 40.0))
    history = np.full((1, T), 1.0)
    sliced = d.slice(history)
    self.assertIsNotNone(sliced.cbounds)
    lo, hi, t_start, t_end = sliced.cbounds[0]
    self.assertAlmostEqual(lo, 5.0 - T)
    self.assertAlmostEqual(hi, 40.0 - T)
    self.assertEqual(t_start, 0)
    self.assertEqual(t_end, N - T)

  def test_cbounds_fully_past_dropped(self):
    d = Device('d', N, (0.0, 2.0), cbounds=[(0.0, 10.0, 0, T)])
    history = np.full((1, T), 0.5)
    sliced = d.slice(history)
    self.assertIsNone(sliced.cbounds)

  def test_cbounds_violation_raises(self):
    d = Device('d', N, (0.0, 2.0), cbounds=(0.0, 3.0))
    history = np.full((1, T), 1.0)
    with self.assertRaises(ValueError):
      d.slice(history)

  def test_slice_and_solve(self):
    d = Device('d', N, (0.5, 1.5))
    x, _ = solve(d, p=1.0)
    history = _history(x)
    sliced = d.slice(history)
    x_rem, meta = solve(sliced, p=1.0)
    self.assertEqual(x_rem.shape, (1, N - T))
    self.assertTrue(meta.success)

  def test_solve_with_history_kwarg(self):
    d = Device('d', N, (0.5, 1.5))
    x, _ = solve(d, p=1.0)
    history = _history(x)
    x_rem1, _ = solve(d.slice(history), p=1.0)
    x_rem2, _ = solve(d, p=1.0, history=history)
    np.testing.assert_allclose(x_rem1, x_rem2, atol=1e-5)

  def test_history_too_long_raises(self):
    d = Device('d', N, (0.0, 1.0))
    with self.assertRaises(ValueError):
      d.slice(np.ones((1, N)))


class TestCDeviceSlice(unittest.TestCase):
  def test_slice_preserves_type(self):
    d = CDevice('c', N, (0.0, 2.0), cbounds=(10.0, 30.0))
    history = np.full((1, T), 0.8)
    sliced = d.slice(history)
    self.assertIsInstance(sliced, CDevice)
    self.assertEqual(len(sliced), N - T)

  def test_slice_and_solve(self):
    d = CDevice('c', N, (0.0, 2.0), cbounds=(5.0, 30.0), a=-0.5)
    x, _ = solve(d, p=0.5)
    history = _history(x, noise=0.05)
    history = np.clip(history, d.lbounds[:T], d.hbounds[:T])
    sliced = d.slice(history)
    x_rem, meta = solve(sliced, p=0.5)
    self.assertTrue(meta.success)
    self.assertEqual(x_rem.shape, (1, N - T))


class TestIDeviceSlice(unittest.TestCase):
  def test_slice_preserves_type(self):
    d = IDevice('i', N, (0.5, 2.0), b=2, c=1)
    history = np.full((1, T), 1.0)
    sliced = d.slice(history)
    self.assertIsInstance(sliced, IDevice)
    self.assertEqual(len(sliced), N - T)

  def test_slice_and_solve(self):
    d = IDevice('i', N, (0.5, 2.0), b=2, c=1)
    x, _ = solve(d, p=0.5)
    history = _history(x)
    sliced = d.slice(history)
    _, meta = solve(sliced, p=0.5)
    self.assertTrue(meta.success)

  def test_per_slot_params_sliced(self):
    b_vec = np.linspace(1, 3, N)
    d = IDevice('i', N, (0.5, 2.0), b=b_vec, c=1)
    history = np.full((1, T), 1.0)
    sliced = d.slice(history)
    np.testing.assert_array_equal(np.asarray(sliced.b), b_vec[T:])


class TestIDevice2Slice(unittest.TestCase):
  def test_slice_preserves_type(self):
    d = IDevice2('i2', N, (0.5, 2.0), p_l=-1.0, p_h=-0.1)
    history = np.full((1, T), 1.0)
    sliced = d.slice(history)
    self.assertIsInstance(sliced, IDevice2)
    self.assertEqual(len(sliced), N - T)

  def test_per_slot_params_sliced(self):
    p_h_vec = np.linspace(-0.5, -0.1, N)
    d = IDevice2('i2', N, (0.5, 2.0), p_l=-1.0, p_h=p_h_vec)
    history = np.full((1, T), 1.0)
    sliced = d.slice(history)
    np.testing.assert_array_equal(np.asarray(sliced.p_h), p_h_vec[T:])

  def test_slice_and_solve(self):
    d = IDevice2('i2', N, (0.5, 2.0), p_l=-1.0, p_h=-0.1)
    x, _ = solve(d, p=0.3)
    history = _history(x)
    _, meta = solve(d, p=0.3, history=history)
    self.assertTrue(meta.success)


class TestGDeviceSlice(unittest.TestCase):
  def test_slice_scalar_cost_coeffs(self):
    d = GDevice('g', N, (-5.0, 0.0), cost_coeffs=[1.0, 0.5, 0.0])
    history = np.full((1, T), -1.5)
    sliced = d.slice(history)
    self.assertIsInstance(sliced, GDevice)
    self.assertEqual(len(sliced), N - T)

  def test_slice_per_slot_cost_coeffs(self):
    coeffs = np.tile([1.0, 0.5, 0.0], (N, 1))
    coeffs[:, 0] += np.linspace(0, 1, N)
    d = GDevice('g', N, (-5.0, 0.0), cost_coeffs=coeffs)
    history = np.full((1, T), -1.0)
    sliced = d.slice(history)
    np.testing.assert_array_equal(sliced.cost_coeffs, coeffs[T:])

  def test_slice_and_solve(self):
    d = GDevice('g', N, (-5.0, 0.0), cost_coeffs=[1.0, 0.5, 0.0])
    x, _ = solve(d, p=0.8)
    history = _history(x, noise=0.1)
    history = np.clip(history, d.lbounds[:T], d.hbounds[:T])
    _, meta = solve(d, p=0.8, history=history)
    self.assertTrue(meta.success)


class TestSDeviceSlice(unittest.TestCase):
  def _make(self, **kwargs):
    defaults = dict(id='s', length=N, bounds=(-5.0, 5.0),
                    c1=0.5, capacity=50.0, start=0.5)
    defaults.update(kwargs)
    return SDevice(**defaults)

  def test_slice_advances_start_soc(self):
    d = self._make()
    history = np.full((1, T), 1.0)
    sliced = d.slice(history)
    self.assertIsInstance(sliced, SDevice)
    self.assertEqual(len(sliced), N - T)
    expected_start = (0.5 * 50.0 + T * 1.0) / 50.0
    self.assertAlmostEqual(sliced.start, expected_start)

  def test_slice_discharge_history(self):
    d = self._make(start=0.8)
    history = np.full((1, T), -1.0)
    sliced = d.slice(history)
    expected_start = (0.8 * 50.0 - T * 1.0) / 50.0
    self.assertAlmostEqual(sliced.start, expected_start)

  def test_infeasible_history_raises(self):
    d = self._make(start=0.0)
    history = np.full((1, T), -2.0)
    with self.assertRaises(ValueError):
      d.slice(history)

  def test_slice_and_solve(self):
    d = self._make()
    x, _ = solve(d, p=0.0)
    history = _history(x, noise=0.1)
    history = np.clip(history, d.lbounds[:T], d.hbounds[:T])
    sliced = d.slice(history)
    x_rem, meta = solve(sliced, p=0.0)
    self.assertTrue(meta.success)
    self.assertEqual(x_rem.shape, (1, N - T))

  def test_full_reconstruct_conserves_capacity(self):
    d = self._make()
    x, _ = solve(d, p=0.0)
    history = _history(x)
    sliced = d.slice(history)
    x_rem, meta = solve(sliced, p=0.0)
    self.assertTrue(meta.success)
    soc_trajectory = sliced.charge_at(x_rem.flatten())
    self.assertTrue((soc_trajectory >= -1e-5).all())
    self.assertTrue((soc_trajectory <= sliced.capacity + 1e-5).all())

  def test_solve_with_history_kwarg(self):
    d = self._make()
    x, _ = solve(d, p=0.0)
    history = _history(x)
    x_rem1, _ = solve(d.slice(history), p=0.0)
    x_rem2, _ = solve(d, p=0.0, history=history)
    np.testing.assert_allclose(x_rem1, x_rem2, atol=1e-5)


class TestTDeviceSlice(unittest.TestCase):
  def _make(self):
    t_ext = 15.0 + 5.0 * np.sin(np.linspace(0, np.pi, N))
    return TDevice(
        id='t', length=N, bounds=(0.0, 3.0),
        sustainment=0.8, efficiency=2.0,
        t_init=20.0, t_optimal=22.0, t_range=3.0,
        t_external=t_ext, c=1,
    )

  def test_slice_length_and_type(self):
    d = self._make()
    history = np.full((1, T), 1.0)
    sliced = d.slice(history)
    self.assertIsInstance(sliced, TDevice)
    self.assertEqual(len(sliced), N - T)

  def test_slice_advances_t_init(self):
    d = self._make()
    history = np.full((1, T), 1.0)
    from device_kit.utils import soc
    t_base_T = d.t_base[:T]
    expected_terminal_t = (t_base_T + soc(history.flatten(), s=d.sustainment, e=d.efficiency))[-1]
    sliced = d.slice(history)
    self.assertAlmostEqual(sliced.t_init, expected_terminal_t)

  def test_t_external_sliced(self):
    d = self._make()
    history = np.full((1, T), 0.5)
    sliced = d.slice(history)
    np.testing.assert_array_equal(sliced.t_external, np.asarray(d.t_external)[T:])

  def test_slice_and_solve(self):
    d = self._make()
    x, _ = solve(d, p=0.3)
    history = _history(x, noise=0.05)
    history = np.clip(history, d.lbounds[:T], d.hbounds[:T])
    _, meta = solve(d, p=0.3, history=history)
    self.assertTrue(meta.success)


class TestDeviceSetSlice(unittest.TestCase):
  def _make_model(self):
    np.random.seed(7)
    return DeviceSet('site', [
        Device('load', N, (0.5, 2.0)),
        IDevice2('flex', N, (0.3, 1.5), p_l=-1.0, p_h=-0.2),
        SDevice('batt', N, (-3.0, 3.0), c1=0.3, capacity=20.0, start=0.5),
        GDevice('gen', N, (-5.0, 0.0), cost_coeffs=[1.0, 0.5, 0.0]),
    ], sbounds=(0.0, 0.0))

  def test_slice_length_and_shape(self):
    model = self._make_model()
    x, _ = solve(model, p=0.0)
    history = _history(x)
    sliced = model.slice(history)
    self.assertEqual(len(sliced), N - T)
    self.assertEqual(sliced.shape, (model.shape[0], N - T))

  def test_sbounds_sliced(self):
    model = self._make_model()
    x, _ = solve(model, p=0.0)
    history = _history(x)
    sliced = model.slice(history)
    np.testing.assert_array_equal(sliced.sbounds, model.sbounds[T:])

  def test_children_sliced(self):
    model = self._make_model()
    x, _ = solve(model, p=0.0)
    history = _history(x)
    sliced = model.slice(history)
    for child in sliced.devices:
      self.assertEqual(len(child), N - T)

  def test_slice_and_solve(self):
    model = self._make_model()
    x, _ = solve(model, p=0.0)
    history = _history(x, noise=0.05)
    for i, (d, (offset, nrows)) in enumerate(zip(model.devices, model.partition)):
      row = slice(offset, offset + nrows)
      history[row, :] = np.clip(history[row, :], d.lbounds[:T], d.hbounds[:T])
    x_rem, meta = solve(model, p=0.0, history=history)
    self.assertTrue(meta.success)
    self.assertEqual(x_rem.shape, (model.shape[0], N - T))
    np.testing.assert_allclose(x_rem.sum(axis=0), 0.0, atol=1e-4)

  def test_nested_deviceset(self):
    inner = DeviceSet('inner', [
        Device('a', N, (0.0, 1.0)),
        Device('b', N, (0.0, 1.0)),
    ], sbounds=(0.5, 1.5))
    outer = DeviceSet('outer', [
        inner,
        GDevice('gen', N, (-3.0, 0.0), cost_coeffs=[1.0, 0.5, 0.0]),
    ], sbounds=(0.0, 0.0))
    x, _ = solve(outer, p=0.0)
    history = _history(x)
    sliced = outer.slice(history)
    self.assertEqual(len(sliced), N - T)
    _, meta = solve(sliced, p=0.0)
    self.assertTrue(meta.success)


class TestSubBalancedDeviceSetSlice(unittest.TestCase):
  def test_slice_preserves_labels(self):
    model = SubBalancedDeviceSet('site', [
        Device('load-A', N, (0.0, 2.0)),
        Device('gen-A', N, (-2.0, 0.0)),
        Device('load-B', N, (0.0, 1.5)),
    ], labels=['A'])
    x, _ = solve(model, p=0.0)
    history = _history(x)
    sliced = model.slice(history)
    self.assertIsInstance(sliced, SubBalancedDeviceSet)
    self.assertEqual(sliced.labels, model.labels)
    self.assertEqual(sliced.constraint_type, model.constraint_type)
    self.assertEqual(len(sliced), N - T)


class TestWindowDeviceSlice(unittest.TestCase):
  def test_slice_preserves_type(self):
    d = WindowDevice('w', N, (0.0, 2.0), w=4, c=0.5)
    history = np.full((1, T), 0.8)
    sliced = d.slice(history)
    self.assertIsInstance(sliced, WindowDevice)
    self.assertEqual(len(sliced), N - T)
    self.assertEqual(sliced.w, d.w)
    self.assertEqual(sliced.c, d.c)

  def test_slice_and_solve(self):
    d = WindowDevice('w', N, (0.1, 2.0), w=6, c=0.3)
    x, _ = solve(d, p=0.2)
    history = _history(x, noise=0.05)
    history = np.clip(history, d.lbounds[:T], d.hbounds[:T])
    _, meta = solve(d, p=0.2, history=history)
    self.assertTrue(meta.success)


class TestADeviceSlice(unittest.TestCase):
  def test_raises_not_implemented(self):
    from device_kit.functions import NullFunction
    d = ADevice('a', N, (0.0, 1.0), f=NullFunction())
    with self.assertRaises(NotImplementedError):
      d.slice(np.zeros((1, T)))


class TestSolveHistoryKwarg(unittest.TestCase):
  def test_price_vector_sliced(self):
    d = Device('d', N, (0.0, 2.0))
    p_full = np.linspace(0.1, 1.0, N)
    history = np.full((1, T), 1.0)
    x1, _ = solve(d.slice(history), p=p_full[T:])
    x2, _ = solve(d, p=p_full, history=history)
    np.testing.assert_allclose(x1, x2, atol=1e-5)

  def test_scalar_price_unchanged(self):
    d = Device('d', N, (0.5, 1.5))
    history = np.full((1, T), 1.0)
    x, meta = solve(d, p=0.5, history=history)
    self.assertTrue(meta.success)
    self.assertEqual(x.shape, (1, N - T))


if __name__ == "__main__":
  unittest.main()
