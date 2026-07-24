import os

from loguru import logger

__version__ = "0.5.0"

# Disable torchquad's own log records by default so importing the library never
# adds output to a host application's loguru configuration (issue #184).
# set_log_level() re-enables them when the user opts into logging.
logger.disable("torchquad")

# Set for release builds
TORCHQUAD_DISABLE_LOGGING = True

# TODO: Currently this is the way to expose to the docs
# hopefully changes with setup.py
from .integration.integration_grid import IntegrationGrid
from .integration.monte_carlo import MonteCarlo
from .integration.trapezoid import Trapezoid
from .integration.simpson import Simpson
from .integration.boole import Boole
from .integration.vegas import VEGAS
from .integration.gaussian import GaussLegendre
from .integration.gaussian import Gaussian
from .integration.grid_integrator import GridIntegrator
from .integration.base_integrator import BaseIntegrator

from .integration.rng import RNG
from .integration.qmc import Sobol


from .utils.set_log_level import set_log_level
from .utils.enable_cuda import enable_cuda
from .utils.set_precision import set_precision
from .utils.set_up_backend import set_up_backend
from .utils.deployment_test import _deployment_test

__all__ = [
    "__version__",
    "GridIntegrator",
    "BaseIntegrator",
    "IntegrationGrid",
    "MonteCarlo",
    "Trapezoid",
    "Simpson",
    "Boole",
    "VEGAS",
    "GaussLegendre",
    "Gaussian",
    "RNG",
    "Sobol",
    "enable_cuda",
    "set_precision",
    "set_log_level",
    "set_up_backend",
    "_deployment_test",
]

if not TORCHQUAD_DISABLE_LOGGING:
    set_log_level(os.environ.get("TORCHQUAD_LOG_LEVEL", "WARNING"))
    logger.info("Initializing torchquad.")
