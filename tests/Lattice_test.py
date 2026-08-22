"""Tests for the rank-1 lattice quasi-Monte Carlo sampler.

Lattice points plugged into ``MonteCarlo`` via the ``rng`` slot must (a) integrate
smooth functions accurately, (b) beat plain pseudo-random Monte Carlo at the same
sample count, and (c) reproduce bit-for-bit for a fixed seed -- the same three
properties covered for ``Sobol`` in ``test_sobol.py``, reused here almost verbatim.

Unlike ``Sobol``, ``Lattice`` only supports the torch backend: it relies directly
on torch tensors and ``torch.Generator`` with no fallback path for other
backends, so its ``backend`` constructor argument only accepts ``"torch"``
(anything else raises ``ValueError``). All coverage below runs on torch only;
there is no per-backend loop.

Like ``Sobol.uniform``, ``Lattice.uniform`` casts its ``size`` components with
plain ``int(...)``, relying on the fact that the only real caller
(``MonteCarlo.calculate_sample_points``) always passes an already-validated
``N`` (checked by ``BaseIntegrator._check_inputs``) and a ``dim`` derived from
``integration_domain.shape[0]``, which is always a clean Python int. There is
therefore no dedicated test here for malformed ``size`` components (e.g. bools
or non-whole floats); ``Sobol``'s test suite does not cover this either, for
the same reason.

The remaining tests target properties that are specific to the rank-1 lattice
construction and have no Sobol equivalent: the closed-form point formula, the
one-shot random shift (drawn once and reused for the sampler's lifetime), and
input validation on N and dim.
"""

import numpy as np
import pytest
import torch

from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.qmc import (
    Lattice,
    load_lattice_vector,
    _MIN_POINTS,
    _MAX_POINTS,
    _max_dimension,
)
from torchquad.utils.set_up_backend import set_up_backend
from helper_functions import setup_test_for_backend

# Same smooth, non-separable-into-a-polynomial integrand as the Sobol tests:
# prod_i cos(pi/2 * x_i) over [0, 1]^dim integrates to (2/pi)^dim. QMC should
# shine here.
_DIM = 3
_N = (
    2**13
)  # 8192 points: a power of 2, inside the [_MIN_POINTS, _MAX_POINTS] range Lattice requires
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


# ---------------------------------------------------------------------------
# Tests reused from the Sobol suite: accuracy, beating plain MC, determinism,
# and gradient flow. The backend loop collapses to "torch" only.
# ---------------------------------------------------------------------------


def _lattice_accuracy_test(backend, dtype_name=None):
    """Lattice MC must integrate a smooth function close to the analytic value."""
    mc = MonteCarlo()
    result = _to_float(
        mc.integrate(
            _integrand,
            dim=_DIM,
            N=_N,
            integration_domain=_DOMAIN,
            rng=Lattice(backend="torch", seed=0),
        )
    )
    assert abs(result - _EXPECTED) < 1e-3, (
        f"Lattice result {result} too far from analytic {_EXPECTED}"
    )


def _lattice_beats_mc_test(backend, dtype_name=None):
    """At equal N, Lattice must be more accurate than pseudo-random Monte Carlo.

    Both use a fixed seed, so this comparison is deterministic (no flakiness), and
    the QMC error is orders of magnitude smaller for a smooth integrand.
    """
    mc = MonteCarlo()
    lattice_error = abs(
        _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Lattice(backend="torch", seed=0),
            )
        )
        - _EXPECTED
    )
    mc_error = abs(
        _to_float(mc.integrate(_integrand, dim=_DIM, N=_N, integration_domain=_DOMAIN, seed=0))
        - _EXPECTED
    )
    assert lattice_error < mc_error, (
        f"Lattice error {lattice_error} not below Monte Carlo error {mc_error}"
    )


def _lattice_determinism_test(backend, dtype_name=None):
    """A fixed seed must reproduce the same result bit-for-bit, across fresh
    ``Lattice`` instances (the shift is drawn once per instance, from ``seed``)."""
    mc = MonteCarlo()

    def run():
        return _to_float(
            mc.integrate(
                _integrand,
                dim=_DIM,
                N=_N,
                integration_domain=_DOMAIN,
                rng=Lattice(backend="torch", seed=42),
            )
        )

    assert run() == run(), "Lattice integration is not reproducible for a fixed seed"


test_lattice_accuracy_torch = setup_test_for_backend(_lattice_accuracy_test, "torch", "float64")
test_lattice_beats_mc_torch = setup_test_for_backend(_lattice_beats_mc_test, "torch", "float64")
test_lattice_determinism_torch = setup_test_for_backend(
    _lattice_determinism_test, "torch", "float64"
)


def test_lattice_preserves_gradient():
    """Lattice points are constants, so autodiff through the integral must survive."""
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
        rng=Lattice(backend="torch", seed=0),
    )
    result.backward()
    assert abs(_to_float(parameter.grad) - 0.5) < 1e-3, (
        "Gradient did not flow through the Lattice-sampled Monte Carlo integral"
    )


# ---------------------------------------------------------------------------
# Lattice-specific tests: backend validation, the closed-form construction,
# the one-shot shift, and input validation on N and dim. None of these have a
# Sobol equivalent.
# ---------------------------------------------------------------------------


def test_lattice_rejects_non_torch_backend():
    with pytest.raises(ValueError, match="torch"):
        Lattice(backend="numpy")


def test_lattice_matches_closed_form_without_shift():
    """Unshifted points must equal frac(i * z / N) exactly, for the actual
    generating vector on disk. This pins down the construction itself,
    independently of the downstream integration-accuracy tests above.
    """
    set_up_backend("torch", "float64")
    dim = 4
    n = 1024  # smallest supported N, keeps the manual computation cheap

    z = load_lattice_vector(d=dim).astype(np.int64)
    i = np.arange(n, dtype=np.int64)
    expected = ((i[:, None] * z[None, :]) % n) / n

    points = Lattice(backend="torch", shift=False).uniform([n, dim], torch.float64).cpu().numpy()

    np.testing.assert_allclose(
        points,
        expected,
        atol=1e-12,
        err_msg="Unshifted Lattice points do not match the closed-form frac(i*z/N) formula",
    )


def test_lattice_same_seed_reproduces_shift_across_calls():
    """The shift is redrawn on every call to ``uniform``, not cached -- but with
    a fixed seed, each redraw must reproduce the same values, so successive
    calls on the same instance return identical points."""
    set_up_backend("torch", "float64")
    lattice = Lattice(backend="torch", seed=0, shift=True)

    first = lattice.uniform([1024, 2], torch.float64).clone()
    second = lattice.uniform([1024, 2], torch.float64)

    assert torch.equal(first, second), (
        "Calling uniform() twice with the same seed must return identical points, "
        "since the seeded shift draw is reproducible"
    )


def test_lattice_unseeded_shift_differs_across_calls():
    """Without a seed, each call to ``uniform`` draws its own independent
    shift, so successive calls on the same instance must return different
    points."""
    set_up_backend("torch", "float64")
    lattice = Lattice(backend="torch", shift=True)  # no seed

    first = lattice.uniform([1024, 2], torch.float64).clone()
    second = lattice.uniform([1024, 2], torch.float64)

    assert not torch.equal(first, second), (
        "Calling uniform() twice without a seed should not return identical "
        "points, since each call draws its own independent shift"
    )


def test_lattice_shift_changes_points():
    """With shift=True, the returned points must differ from the unshifted
    lattice -- otherwise the shift would be silently having no effect."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    unshifted = Lattice(backend="torch", shift=False).uniform([n, dim], torch.float64)
    shifted = Lattice(backend="torch", seed=0, shift=True).uniform([n, dim], torch.float64)

    assert not torch.equal(unshifted, shifted), (
        "Shifted points must not be identical to the unshifted lattice"
    )


def test_lattice_different_seeds_give_different_shifts():
    """Two instances with different seeds must (almost surely) produce different
    point sets, since each draws its own independent random shift."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    points_a = Lattice(backend="torch", seed=1, shift=True).uniform([n, dim], torch.float64)
    points_b = Lattice(backend="torch", seed=2, shift=True).uniform([n, dim], torch.float64)

    assert not torch.equal(points_a, points_b), (
        "Different seeds should not produce the same shifted lattice"
    )


def test_lattice_seed_ignored_when_shift_false():
    """When shift=False, the seed argument must have no effect: unshifted
    lattice points depend only on N and dim, never on the seed."""
    set_up_backend("torch", "float64")
    n, dim = 1024, 3

    points_a = Lattice(backend="torch", seed=1, shift=False).uniform([n, dim], torch.float64)
    points_b = Lattice(backend="torch", seed=99, shift=False).uniform([n, dim], torch.float64)

    assert torch.equal(points_a, points_b), "Unshifted lattice points must not depend on the seed"


def test_lattice_points_lie_in_unit_cube():
    """Every coordinate must lie in [0, 1), both with and without a shift."""
    set_up_backend("torch", "float64")
    n, dim = 2048, 5

    for shift in (False, True):
        points = Lattice(backend="torch", seed=0, shift=shift).uniform([n, dim], torch.float64)
        assert torch.all(points >= 0) and torch.all(points < 1), (
            f"Lattice points (shift={shift}) fall outside [0, 1)"
        )


def test_lattice_rejects_non_power_of_two_point_count():
    """The simple lattice construction is only valid when N is a power of 2;
    a non-power-of-two N (even if within [_MIN_POINTS, _MAX_POINTS]) must
    raise, since the construction's validity guarantee does not hold
    otherwise."""
    set_up_backend("torch", "float64")
    with pytest.raises(ValueError, match="power of 2"):
        Lattice(backend="torch", shift=False).uniform([1030, 2], torch.float64)


@pytest.mark.parametrize("bad_n", [_MIN_POINTS - 1, _MAX_POINTS + 1, 500, 2_000_000])
def test_lattice_rejects_out_of_range_point_counts(bad_n):
    """N below _MIN_POINTS or above _MAX_POINTS must raise, near the boundary
    or far outside it."""
    with pytest.raises(ValueError):
        Lattice(backend="torch").uniform([bad_n, 2], torch.float64)


def test_lattice_rejects_out_of_range_dimensions():
    """dim below 1 or above the generating vector file's max must raise, near
    the boundary or far outside it.

    Not parametrized: the upper bound is derived from the actual data file
    via ``_max_dimension()``, which must run inside a test function rather
    than at module collection time, to avoid loading the file on import.
    """
    max_dim = _max_dimension()
    for bad_dim in [0, -1, max_dim + 1, max_dim + 10000]:
        with pytest.raises(ValueError):
            Lattice(backend="torch").uniform([1024, bad_dim], torch.float64)


if __name__ == "__main__":
    for _test in (_lattice_accuracy_test, _lattice_beats_mc_test, _lattice_determinism_test):
        _test("torch")
    test_lattice_rejects_non_torch_backend()
    test_lattice_preserves_gradient()
    test_lattice_matches_closed_form_without_shift()
    test_lattice_same_seed_reproduces_shift_across_calls()
    test_lattice_unseeded_shift_differs_across_calls()
    test_lattice_shift_changes_points()
    test_lattice_different_seeds_give_different_shifts()
    test_lattice_seed_ignored_when_shift_false()
    test_lattice_points_lie_in_unit_cube()
    test_lattice_rejects_non_power_of_two_point_count()
    test_lattice_rejects_out_of_range_dimensions()
    print("All Lattice tests passed!")
