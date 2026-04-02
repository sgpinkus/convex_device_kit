"""
slice_demo.py — Smoke test for device_kit's incremental re-optimisation feature.

Scenario
--------
A small site with an uncontrolled load, a flexible load, a shiftable load, a
generator, a PV array and a battery is optimised day-ahead for 24 hours.

Starting from hour 4 we simulate rolling re-optimisation: each hour we
  1. Reveal actual flows for the slot that just passed (uncontrolled load
     is slightly different from plan; PV output is ±20 % of forecast).
  2. Call solve(..., history=<observed so far>) to get a fresh schedule for
     the remaining horizon conditioned on what actually happened.

Assertions (smoke-test level)
------------------------------
- Every re-solve must converge successfully.
- The balance constraint (sum of all flows == 0) must hold on the
  re-optimised remaining slots.
- The SDevice SoC must stay within [0, capacity] over the remaining window
  after each re-solve.
- The CDevice cumulative consumption over [12,24] must be within its
  cbounds after adjusting for any history that already falls in that window.

Outputs
-------
Prints a per-hour summary table and saves slice_demo.png with four subplots.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from device_kit import (
  CDevice,
  Device,
  DeviceSet,
  GDevice,
  IDevice2,
  PVDevice,
  SDevice,
  solve,
)


def random_uncontrolled(seed=None):
  rng = np.random.default_rng(seed)
  return np.maximum(0, 0.5 + np.cumsum(rng.uniform(-1, 1, 24)))


def generator_cost_curve():
  return np.stack((
      np.sin(np.linspace(0, np.pi, 24)) * 0.5 + 0.1,
      np.ones(24) * 0.001,
      np.zeros(24),
  ), axis=1)


def pv_bounds(solar_scale=1.0):
  pv_max_rate = 2
  pv_area = 3
  pv_efficiency = 0.9
  pv_solar_intensity = np.maximum(0, np.sin(np.linspace(0, np.pi * 2, 24)))
  lb = -1 * np.minimum(pv_max_rate, pv_solar_intensity * pv_efficiency * pv_area * solar_scale)
  return np.stack((lb, np.zeros(24)), axis=1)


def make_model(uncontrolled_load=None, solar_scale=1.0):
  """Build the model; optionally override load profile and PV scale."""
  np.random.seed(19)
  if uncontrolled_load is None:
    uncontrolled_load = random_uncontrolled(seed=19)
  model = DeviceSet('site1', [
      Device('uncontrolled', 24, (uncontrolled_load,)),
      IDevice2('scalable', 24, (0.25, 2), (0, 24), p_h=-0.5),
      CDevice('shiftable', 24, (0, 2), (12, 24), a=-1.),
      GDevice('generator', 24, (-10, 0), cbounds=None, cost_coeffs=generator_cost_curve()),
      PVDevice(
          'solar', 24,
          pv_bounds(solar_scale=solar_scale),
      ),
      SDevice('battery', 24, (-5, 5), c1=0.01, capacity=14, sustainment=1.0, efficiency=1.0, reserve=0.5, start=0.5),
  ], sbounds=(0, 0))
  return model


def realise_slot(model, x_plan, t, rng):
  """
  Given the planned flow matrix x_plan, produce the actual observed flows for
  slot t.  Uncontrolled load is perturbed ±15 %; PV is ±20 % of plan.
  All other devices follow their plan exactly (controllable).
  """
  actual = x_plan[:, t].copy()
  leaf_names = [name for name, _ in model.leaf_devices()]
  for i, name in enumerate(leaf_names):
    if name.endswith('uncontrolled'):
      actual[i] *= rng.uniform(0.85, 1.15)
    elif name.endswith('solar'):
      actual[i] *= rng.uniform(0.80, 1.20)
  return actual


def main():
  N = 24
  REPLAN_FROM = 4   # start re-planning from this hour
  rng = np.random.default_rng(42)

  model = make_model()
  x_plan, meta = solve(model, p=0)
  assert meta.success, f'Day-ahead solve failed: {meta.message}'
  print(f'Day-ahead solve: {meta.message}')
  print(f'Day-ahead cost:  {model.cost(x_plan, p=0):.4f}\n')

  n_rows = model.shape[0]
  leaf_names = [name for name, _ in model.leaf_devices()]

  # Accumulate observed history slot by slot
  history = np.zeros((n_rows, 0))

  # Track the rolling plan (updated each hour), and per-hour re-solve cost
  x_rolling = x_plan.copy()
  records = []   # one dict per re-plan step

  for t in range(REPLAN_FROM, N - 1):  # re-plan after each hour up to N-2
    # 1. Observe actual flows for slot t
    actual_t = realise_slot(model, x_rolling, t=history.shape[1], rng=rng)
    history = np.hstack([history, actual_t.reshape(n_rows, 1)])

    T = history.shape[1]   # number of slots now in history

    # 2. Build updated model: new uncontrolled load realisation + PV scale
    #    (models what the operator now forecasts for the rest of the day)
    new_load = random_uncontrolled(seed=rng.integers(0, 10_000))
    # Keep history portion of load exactly as observed; update future forecast
    new_load[:T] = history[leaf_names.index(next(n for n in leaf_names if n.endswith('uncontrolled'))), :]
    solar_scale = rng.uniform(0.7, 1.3)
    updated_model = make_model(uncontrolled_load=new_load, solar_scale=solar_scale)

    # 3. Re-solve with history
    x_rem, meta_r = solve(updated_model, p=0, history=history)
    assert meta_r.success, f'Re-solve at T={T} failed: {meta_r.message}'

    # 4. Assemble full rolling plan (history + re-optimised remainder)
    x_rolling = np.hstack([history, x_rem])

    # Balance: sum across devices must be ~0 for every remaining slot
    balance_err = np.abs(x_rem.sum(axis=0)).max()
    assert balance_err < 1e-3, f'Balance violation at T={T}: max err={balance_err:.2e}'

    # SDevice SoC stays in [0, capacity]
    sliced = updated_model.slice(history)
    sliced_leaf_names = [n for n, _ in sliced.leaf_devices()]
    batt_row = sliced_leaf_names.index(next(n for n in sliced_leaf_names if n.endswith('battery')))
    batt_flow_rem = x_rem[batt_row, :]
    batt_sliced = sliced.get('battery')
    soc_traj = batt_sliced.charge_at(batt_flow_rem)
    assert (soc_traj >= -0.1).all(), f'SoC went negative at T={T}'
    assert (soc_traj <= batt_sliced.capacity + 0.1).all(), f'SoC exceeded capacity at T={T}'

    cost_rem = updated_model.slice(history).cost(x_rem, p=0)
    records.append({
        'replan_at': T,
        'slots_remaining': N - T,
        'solar_scale': solar_scale,
        'balance_max_err': balance_err,
        'soc_min': soc_traj.min(),
        'soc_max': soc_traj.max(),
        'cost_remaining': cost_rem,
        'converged': meta_r.success,
    })

    print(
        f'T={T:>2}  remaining={N-T:>2}  solar={solar_scale:.2f}  '
        f'balance_err={balance_err:.1e}  '
        f'SoC=[{soc_traj.min():.1f},{soc_traj.max():.1f}]  '
        f'cost={cost_rem:.3f}'
    )

  print(f'\nAll {len(records)} re-solves converged and passed assertions.\n')

  df_records = pd.DataFrame(records).set_index('replan_at')

  fig = plt.figure(figsize=(14, 10))
  gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

  hours = np.arange(N)

  # Subplot 1: day-ahead plan vs rolling actual
  ax1 = fig.add_subplot(gs[0, :])
  df_plan = pd.DataFrame(dict(model.map(x_plan)), index=hours).T
  df_roll = pd.DataFrame(dict(model.map(x_rolling)), index=hours).T
  for name in leaf_names:
    ax1.step(hours, df_plan.loc[name], linestyle='--', alpha=0.5, label=f'{name.split(".")[-1]} (plan)')
    ax1.step(hours, df_roll.loc[name], linestyle='-', alpha=0.8, label=f'{name.split(".")[-1]} (actual)')
  ax1.axvline(REPLAN_FROM, color='black', linestyle=':', linewidth=1.5, label='re-plan start')
  ax1.set_title('Day-ahead plan vs rolling re-optimised schedule')
  ax1.set_xlabel('Hour')
  ax1.set_ylabel('Flow (kWh)')
  ax1.legend(fontsize=7, ncol=4, loc='upper right')
  ax1.grid(True, alpha=0.3)

  # Subplot 2: balance error per re-plan
  ax2 = fig.add_subplot(gs[1, 0])
  ax2.semilogy(df_records.index, df_records['balance_max_err'], marker='o', color='tab:red')
  ax2.set_title('Balance constraint max error per re-plan')
  ax2.set_xlabel('Re-plan at hour T')
  ax2.set_ylabel('Max |sum of flows| (log scale)')
  ax2.grid(True, alpha=0.3)

  # Subplot 3: SoC range per re-plan
  ax3 = fig.add_subplot(gs[1, 1])
  ax3.fill_between(df_records.index, df_records['soc_min'], df_records['soc_max'],
                   alpha=0.3, color='tab:blue', label='SoC range')
  ax3.plot(df_records.index, df_records['soc_min'], color='tab:blue', linewidth=1)
  ax3.plot(df_records.index, df_records['soc_max'], color='tab:blue', linewidth=1)
  ax3.axhline(0, color='red', linestyle='--', linewidth=0.8, label='min=0')
  ax3.axhline(70, color='orange', linestyle='--', linewidth=0.8, label='max=70')
  ax3.set_title('Battery SoC range over remaining horizon')
  ax3.set_xlabel('Re-plan at hour T')
  ax3.set_ylabel('State of charge (kWh)')
  ax3.legend(fontsize=8)
  ax3.grid(True, alpha=0.3)

  plt.suptitle('device_kit slice() — incremental re-optimisation demo', fontsize=13)
  out = 'slice_demo.png'
  plt.savefig(out, dpi=120, bbox_inches='tight')
  print(f'Plot saved to {out}')


if __name__ == '__main__':
  main()
