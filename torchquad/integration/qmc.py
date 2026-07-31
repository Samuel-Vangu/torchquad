import warnings

from autoray import numpy as anp


class Sobol:
    """A scrambled Sobol low-discrepancy sampler, shaped like :class:`RNG`.

    Pass an instance as the ``rng`` argument of :meth:`MonteCarlo.integrate` to
    turn plain Monte Carlo into quasi-Monte Carlo (QMC): Sobol points cover the
    unit hypercube far more evenly than pseudo-random draws, so for smooth
    integrands the error shrinks close to ``O(1/N)`` instead of the ``O(1/sqrt(N))``
    of plain Monte Carlo.

    Like :class:`RNG`, an instance exposes ``uniform(size, dtype)`` returning
    points in ``[0, 1)`` as a backend tensor, so it is a drop-in replacement for
    the sampler ``MonteCarlo`` uses internally.

    Points are generated with PyTorch's ``torch.quasirandom.SobolEngine`` for the
    ``torch`` backend and with ``scipy.stats.qmc.Sobol`` for the others (converted
    to the requested backend). As with plain Monte Carlo the sample points are
    constants, so gradients still flow through the integrand and the integration
    domain; only the point *placement* differs.

    Notes:
        - The number of points should be a power of two for the Sobol balance
          properties to hold; SciPy emits a warning otherwise.
        - This sampler targets the eager :meth:`MonteCarlo.integrate` path, not
          the JIT-compiled one (which builds its own RNG internally).
        - Per-backend Sobol implementations use different scrambling, so results
          are reproducible for a fixed ``seed`` within a backend but do not match
          bit-for-bit across backends.
    """

    def __init__(self, backend, seed=None, scramble=True):
        """Initialize a Sobol sampler.

        Args:
            backend (string): Numerical backend, e.g. "torch". Must match the
                backend of the integration domain it will be used with.
            seed (int or None, optional): Seed for the scrambling. If None, the
                scrambling is randomised. Defaults to None.
            scramble (bool, optional): Whether to apply Owen scrambling, which
                randomises the sequence while preserving its low discrepancy and
                yields an unbiased estimator. Defaults to True.
        """
        self._backend = backend
        self._seed = seed
        self._scramble = scramble

    def uniform(self, size, dtype):
        """Draw Sobol points in ``[0, 1)``.

        Args:
            size (list): Two-element ``[number_of_points, dim]`` shape.
            dtype (backend dtype): Floating point dtype of the returned tensor.

        Returns:
            backend tensor: ``[number_of_points, dim]`` Sobol points in ``[0, 1)``.
        """
        number_of_points, dim = int(size[0]), int(size[1])

        # Warn on non-power-of-two counts uniformly across backends: SciPy already
        # warns on its own path, but torch's SobolEngine would silently return a
        # sequence prefix whose balance properties do not hold.
        if self._backend == "torch" and number_of_points & (number_of_points - 1) != 0:
            warnings.warn(
                "The balance properties of Sobol points require the number of "
                f"points to be a power of 2, but {number_of_points} was requested.",
                stacklevel=2,
            )

        if self._backend == "torch":
            # SobolEngine keeps the points as native torch tensors (mirrors how
            # RNG special-cases torch); no NumPy round-trip on the GPU.
            import torch

            engine = torch.quasirandom.SobolEngine(
                dimension=dim, scramble=self._scramble, seed=self._seed
            )
            points = engine.draw(number_of_points, dtype=dtype)
            # SobolEngine always draws on the CPU; move the points onto the
            # current default device (e.g. CUDA) so they line up with the
            # integration domain, matching what torch.rand does in RNG.
            return points.to(torch.empty(0).device)

        # numpy / jax / tensorflow: generate with SciPy (a hard dependency) and
        # move the constant points onto the requested backend.
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=dim, scramble=self._scramble, seed=self._seed)
        points = sampler.random(number_of_points)
        return anp.array(points, dtype=dtype, like=self._backend)


import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def load_lattice_vector(d, filename="lattice-33002-1024-1048576.9125"):
    """
    Load the first d components of the generating vector z
    from a generating vectors file for rank-1 lattice rules.


    The file must contain two columns:
        dimension   z_j

    Parameters
    ----------
    d : int
        Desired dimension.
    filename : str
        Path to the generating vectors file.

    Returns
    -------
    z : np.ndarray
        NumPy array [z1, z2, ..., zd] with dtype uint64.
    """
    path = DATA_DIR / filename
    if d <= 0:
        raise ValueError("The dimension d must be strictly positive.")

    data = np.loadtxt(path, dtype=np.uint64)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("The file must contain at least two columns: dimension and z_j.")

    max_d = data.shape[0]

    if d > max_d:
        raise ValueError(
            f"Requested dimension d={d}, but the file contains only {max_d} components."
        )

    z = data[:d, 1]

    return z


import numbers


def _check_positive_whole_number(value, name):
    """Accept any real number equal to a whole number (2, 2.0, np.float64(2.0), ...),
    reject bool (a subtype of int in Python) and anything non-integral (2.5)."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, but got a bool.")
    if not isinstance(value, (numbers.Real, np.floating, np.integer)):
        raise TypeError(f"{name} must be a real number, but got {type(value).__name__}.")
    if float(value) != int(value):
        raise ValueError(f"{name} must be a whole number, but got {value}.")
    return int(value)


class Lattice:
    """A randomly shifted rank-1 lattice low-discrepancy sampler.

    Pass an instance as the ``rng`` argument of :meth:`MonteCarlo.integrate` to
    turn plain Monte Carlo into quasi-Monte Carlo (QMC). Rank-1 lattice points
    cover the unit hypercube more evenly than pseudo-random draws, which can
    reduce the integration error substantially for sufficiently smooth
    integrands.

    Like :class:`RNG`, an instance exposes ``uniform(size, dtype)`` returning
    points in ``[0, 1)`` as a PyTorch tensor, so it is a drop-in replacement for
    the sampler ``MonteCarlo`` uses internally.

    The generating vector is loaded from the file
    ``lattice-33002-1024-1048576.9125``, obtained from the *Lattice Rule
    Generating Vectors* collection maintained by Frances Y. Kuo. This file
    provides an extensible rank-1 lattice rule for dimensions up to 9125 and
    sample sizes from 1024 to 1048576.

    The generating vector was constructed using a component-by-component (CBC)
    algorithm. The construction minimizes, as much as possible, the
    shift-averaged worst-case integration error in a weighted unanchored Sobolev
    space. The file uses order-3 weights with
    ``Gamma_1 = Gamma_2 = 1`` and ``Gamma_3 = 0.5``.

    For a generating vector ``z`` and ``N`` points, the ``i``-th lattice point
    is computed componentwise as

    ``x_i = frac(i * z / N)``,

    where ``i = 0, ..., N - 1`` and ``frac`` denotes the fractional part. Since
    this is an extensible lattice rule, this simple construction is valid when
    the total number of points is an exact power of two.

    When ``shift=True``, a random shift is sampled uniformly from the unit
    hypercube on every call to ``uniform`` (added modulo one to every lattice
    point). This preserves the lattice structure while randomizing the point
    set, making the resulting randomized QMC estimator unbiased for each call.
    With a fixed ``seed``, the shift drawn on each call is identical, so
    successive calls reproduce the same points; without a seed, each call
    draws its own independent shift.

    As with plain Monte Carlo, the sample points are constants, so gradients
    still flow through the integrand and the integration domain; only the point
    placement differs.

    Notes:
        - The dimension must be a whole number between 1 and 9125, inclusive
          (e.g. ``5`` or ``5.0`` are both accepted; ``5.5`` is not).
        - The number of points must be a whole number between 1024 and
          1048576, inclusive.
        - The number of points should be a power of two for the simple
          construction above to be valid for this extensible lattice rule.
        - This sampler targets the eager :meth:`MonteCarlo.integrate` path, not
          the JIT-compiled one (which builds its own RNG internally).
        - The random shift uses a private ``torch.Generator``, not PyTorch's
          global RNG state, so constructing or calling this sampler never
          affects unrelated random draws elsewhere in your program.
    """

    def __init__(self, seed=None, shift=True):
        """Initialize a rank-1 lattice sampler.

        Args:
            seed (int or None, optional): Seed used to generate the random
                shift, redrawn on every call to ``uniform``. A fixed seed
                makes that draw reproducible across calls; if None, each
                call gets an independent random shift. Defaults to None.
            shift (bool, optional): Whether to apply a random shift modulo one
                to the lattice points. Defaults to True.
        """
        self._seed = seed
        self._shift = shift

    def uniform(self, size, dtype):
        """Draw rank-1 lattice points in ``[0, 1)``.

        Args:
            size (list): Two-element ``[number_of_points, dim]`` shape.
            dtype (torch dtype): Floating-point dtype of the returned tensor.

        Returns:
            torch tensor: ``[number_of_points, dim]`` lattice points in
            ``[0, 1)``.

        Raises:
            TypeError: If the number of points or the dimension is not a
                (whole) number.
            ValueError: If the number of points is not between 1024 and
                1048576, or if the dimension is not between 1 and 9125, or if
                either is not a whole number.
        """
        import torch

        number_of_points = _check_positive_whole_number(size[0], "The number of points")
        dim = _check_positive_whole_number(size[1], "The dimension")

        if not 1024 <= number_of_points <= 1048576:
            raise ValueError(
                "The number of points must be between 1024 and 1048576, "
                f"but got {number_of_points}."
            )

        if not 1 <= dim <= 9125:
            raise ValueError(f"The dimension must be between 1 and 9125, but got {dim}.")

        if number_of_points & (number_of_points - 1) != 0:
            warnings.warn(
                "The simple lattice construction used here requires the number "
                f"of points to be a power of 2, but {number_of_points} was "
                "requested.",
                stacklevel=2,
            )

        device = torch.empty(0).device

        # z and index are kept in INT64 for the whole modular reduction, and
        # only converted to `dtype` at the very last step. Computing
        # (index * z) directly in `dtype` (e.g. float32, as requested by the
        # caller) loses precision catastrophically for realistic generating
        # vector magnitudes (millions) and sample sizes: verified to return
        # completely wrong points (e.g. 0.0 instead of 0.6358) for a
        # representative (i, z, N) triple in float32. int64 comfortably
        # covers the largest possible product for this sampler's documented
        # size limits (N <= 1048576, and generating vector entries typically
        # well under 1e8).
        z = torch.tensor(load_lattice_vector(d=dim), dtype=torch.int64)
        index = torch.arange(number_of_points, dtype=torch.int64, device=device)
        mod = (index[:, None] * z[None, :]) % number_of_points  # (N, dim), exact
        points = mod.to(dtype) / number_of_points

        if self._shift:
            gen = torch.Generator(device=device)
            if self._seed is not None:
                gen.manual_seed(self._seed)
            else:
                gen.seed()  # a fresh torch.Generator() otherwise keeps a fixed internal
                # default seed, which would make every unseeded shift
                # identical instead of independently random
            shift_values = torch.rand(dim, generator=gen, device=device)
            points = torch.remainder(points + shift_values.to(dtype), 1)

        return points.to(dtype)
