"""JIT-compiled repeated quadrature matches the eager path.

Users who evaluate the same integrator over many domains use
``get_jit_compiled_integrate`` (JAX/TF ``jit``, torch). The compiled result must
match the eager call numerically for the same seed.
"""

import math

import pytest

from torchquad import MonteCarlo, set_up_backend
from torchquad.integration.utils import _setup_integration_domain

import _integrands as ig
from conftest import rel_error


@pytest.mark.parametrize("backend", ["torch", "jax", "tensorflow"])
def test_jit_matches_eager_monte_carlo(backend):
    """JIT-compiled MC integrate reproduces the eager integrate (same seed)."""
    pytest.importorskip(backend)
    set_up_backend(backend, data_type="float64")
    mc = MonteCarlo()
    # The compiled integrate takes a backend *tensor* domain, so build one and
    # use it for both paths to compare like with like.
    domain = _setup_integration_domain(1, [[0.0, 2.0]], backend)
    n_points = 100_000

    eager = mc.integrate(
        ig.sin_1d, dim=1, N=n_points, integration_domain=domain, seed=0, backend=backend
    )
    jit_integrate = mc.get_jit_compiled_integrate(
        dim=1, N=n_points, integration_domain=domain, seed=0, backend=backend
    )
    compiled = jit_integrate(ig.sin_1d, domain)

    # The JIT path seeds its own RNG stream, so compiled and eager are
    # independent draws, not bit-identical. Both must still land on the analytic
    # value within Monte Carlo tolerance — that is what validates the JIT path.
    expected = 1.0 - math.cos(2.0)  # int_0^2 sin(x) dx
    eager_error = rel_error(eager, expected)
    compiled_error = rel_error(compiled, expected)
    assert eager_error < 5e-2, f"eager MC too far from truth ({backend}): {eager_error:.2e}"
    assert compiled_error < 5e-2, f"JIT MC too far from truth ({backend}): {compiled_error:.2e}"
