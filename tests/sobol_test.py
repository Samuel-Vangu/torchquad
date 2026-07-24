"""Tests for the Sobol quasi-Monte Carlo sampler.

Sobol points plugged into ``MonteCarlo`` via the ``rng`` slot must (a) integrate
smooth functions accurately, (b) beat plain pseudo-random Monte Carlo at the same
sample count, and (c) reproduce bit-for-bit for a fixed seed. Coverage runs on
every backend.
"""

import numpy as np

from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.qmc import Sobol
from helper_functions import setup_test_for_backend

# A smooth, non-separable-into-a-polynomial integrand: prod_i cos(pi/2 * x_i) over
# [0, 1]^dim integrates to (2/pi)^dim. QMC should shine here.
_DIM = 3
_N = 2**13
_DOMAIN = [[0.0, 1.0]] * _DIM
_EXPECTED = (2.0 / np.pi) ** _DIM


def _to_float(result):
    """Convert a scalar backend tensor (possibly on GPU) to a Python float."""
    if hasattr(result, "cpu"):
        result = result.cpu()
    return float(np.asarray(result))


def _integrand(x):
    from autoray import numpy as anp

    return anp.prod(anp.cos(x * (np.pi / 2.0)), axis=1)


def _sobol_accuracy_test(backend, dtype_name=None):
    """Sobol MC must integrate a smooth function close to the analytic value."""
    mc = MonteCarlo()
    result = _to_float(
        mc.integrate(
            _integrand,
            dim=_DIM,
            N=_N,
            integration_domain=_DOMAIN,
            rng=Sobol(backend=backend, seed=0),
        )
    )
    # Tight enough to fail if Sobol silently degraded to plain-MC quality
    # (~1e-4). The measured error is ~2e-8 on the SciPy backends and ~5e-7 on
    # torch's native SobolEngine; 1e-5 clears both with a comfortable margin.
    assert abs(result - _EXPECTED) < 1e-5, (
        f"Sobol result {result} too far from analytic {_EXPECTED}"
    )


def _sobol_beats_mc_test(backend, dtype_name=None):
    """At equal N, Sobol must be more accurate than pseudo-random Monte Carlo.

    Both use a fixed seed, so this comparison is deterministic (no flakiness), and
    the QMC error is orders of magnitude smaller for a smooth integrand.
    """
    mc = MonteCarlo()
    sobol_error = abs(
        _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Sobol(backend=backend, seed=0),
            )
        )
        - _EXPECTED
    )
    mc_error = abs(
        _to_float(mc.integrate(_integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=0))
        - _EXPECTED
    )
    assert sobol_error < mc_error, (
        f"Sobol error {sobol_error} not below Monte Carlo error {mc_error}"
    )


def _sobol_determinism_test(backend, dtype_name=None):
    """A fixed seed must reproduce the same result bit-for-bit."""
    mc = MonteCarlo()

    def run():
        return _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Sobol(backend=backend, seed=42),
            )
        )

    assert run() == run(), "Sobol integration is not reproducible for a fixed seed"


test_sobol_accuracy_numpy = setup_test_for_backend(_sobol_accuracy_test, "numpy", "float64")
test_sobol_accuracy_torch = setup_test_for_backend(_sobol_accuracy_test, "torch", "float64")
test_sobol_accuracy_tensorflow = setup_test_for_backend(
    _sobol_accuracy_test, "tensorflow", "float64"
)
test_sobol_accuracy_jax = setup_test_for_backend(_sobol_accuracy_test, "jax", "float64")

test_sobol_beats_mc_numpy = setup_test_for_backend(_sobol_beats_mc_test, "numpy", "float64")
test_sobol_beats_mc_torch = setup_test_for_backend(_sobol_beats_mc_test, "torch", "float64")
test_sobol_beats_mc_tensorflow = setup_test_for_backend(
    _sobol_beats_mc_test, "tensorflow", "float64"
)
test_sobol_beats_mc_jax = setup_test_for_backend(_sobol_beats_mc_test, "jax", "float64")

test_sobol_determinism_numpy = setup_test_for_backend(_sobol_determinism_test, "numpy", "float64")
test_sobol_determinism_torch = setup_test_for_backend(_sobol_determinism_test, "torch", "float64")
test_sobol_determinism_tensorflow = setup_test_for_backend(
    _sobol_determinism_test, "tensorflow", "float64"
)
test_sobol_determinism_jax = setup_test_for_backend(_sobol_determinism_test, "jax", "float64")


def test_sobol_preserves_gradient():
    """Sobol points are constants, so autodiff through the integral must survive."""
    import torch

    from torchquad.utils.set_up_backend import set_up_backend

    set_up_backend("torch", "float64")
    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    # integral over [0, 1] of parameter * x is parameter / 2, so d/dparameter = 1/2.
    def parametric(x):
        return parameter * x[:, 0]

    mc = MonteCarlo()
    result = mc.integrate(
        parametric,
        dim=1,
        N=_N,
        integration_domain=[[0.0, 1.0]],
        rng=Sobol(backend="torch", seed=0),
    )
    result.backward()
    assert abs(_to_float(parameter.grad) - 0.5) < 1e-3, (
        "Gradient did not flow through the Sobol-sampled Monte Carlo integral"
    )


if __name__ == "__main__":
    for _backend in ["numpy", "torch", "tensorflow", "jax"]:
        _sobol_accuracy_test(_backend)
        _sobol_beats_mc_test(_backend)
        _sobol_determinism_test(_backend)
    test_sobol_preserves_gradient()
    print("All Sobol tests passed!")
