import numpy as np
import pytest

from RaTag.el_tpc import physics
from RaTag.core.config import DRIFT_VELOCITY_PARAMS


def test_gas_density_cm3_basic():
    P = 2.0  # bar
    T = 300.0  # K
    expected = (P * 1e5) / (physics.k_B * T) * 1e-6
    assert pytest.approx(physics.gas_density_cm3(P, T), rel=1e-12) == expected


def test_compute_reduced_field():
    field = 150.0  # V·cm^-1 (V/cm)
    density = 1e19  # cm^-3
    assert physics.compute_reduced_field(field, density) == pytest.approx(field / density)


def test_transport_saturation_and_redfield_to_speed_inverse():
    rE = 0.5  # Td (within default solver bounds)
    v = physics.redfield_to_speed(rE, params=DRIFT_VELOCITY_PARAMS)
    # invert
    rE_back = physics.speed_to_redfield(v, params=DRIFT_VELOCITY_PARAMS, rE_min=0.01, rE_max=3.0)
    assert pytest.approx(rE_back, rel=1e-6) == rE


def test_drift_curve_vectorized():
    rE_list = [0.1, 0.5, 1.0]
    arr = physics.drift_curve(rE_list, DRIFT_VELOCITY_PARAMS)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == len(rE_list)
    # elementwise equals redfield_to_speed for each entry
    for r, v in zip(rE_list, arr):
        assert pytest.approx(physics.redfield_to_speed(r, DRIFT_VELOCITY_PARAMS), rel=1e-12) == v


def test_longitudinal_diffusion_and_sigma():
    rE = 1.0
    a, b = 2.0, 0.1
    D = physics.longitudinal_diffusion_coeff(rE, a, b)
    assert pytest.approx(D, rel=1e-12) == a / np.sqrt(rE) + b

    t = 10.0  # µs
    sigma = physics.diffusion_sigma(D, t)
    assert pytest.approx(sigma, rel=1e-12) == np.sqrt(2.0 * D * t)


def test_s2_pulse_width_consistency():
    z = 10.0  # mm
    rE = 0.5
    diff_params = {"a": 1.0, "b": 0.0}
    # use the default DRIFT_VELOCITY_PARAMS for consistency with redfield->speed
    sigma_t = physics.s2_pulse_width(z, rE, DRIFT_VELOCITY_PARAMS, diff_params)

    # compute expected pieces explicitly
    v_d = physics.redfield_to_speed(rE, DRIFT_VELOCITY_PARAMS)  # mm/µs
    t_d = z / v_d
    D_L = physics.longitudinal_diffusion_coeff(rE, **diff_params)
    sigma_z = physics.diffusion_sigma(D_L, t_d)
    expected = sigma_z / v_d

    assert pytest.approx(sigma_t, rel=1e-12) == expected