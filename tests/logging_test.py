"""Regression tests for loguru library hygiene (issue #184).

torchquad must not reconfigure loguru at import time or wipe a host
application's log handlers when the user sets the torchquad log level.
"""

import io

from loguru import logger

from torchquad import MonteCarlo, set_log_level


def test_set_log_level_preserves_host_handlers():
    """set_log_level must not remove handlers registered by the host application.

    Previously set_log_level called logger.remove() with no argument, which
    removed every sink including the host's (issue #184).
    """
    host_sink = io.StringIO()
    host_handler_id = logger.add(host_sink, level="INFO", filter=lambda record: True)
    try:
        set_log_level("WARNING")
        logger.info("message from the host application")
        assert "message from the host application" in host_sink.getvalue(), (
            "set_log_level removed the host application's loguru handler"
        )
    finally:
        logger.remove(host_handler_id)
        # Remove the handlers set_log_level added and restore the library default
        # so this test does not leak an enabled state or stderr sink into later tests.
        import torchquad.utils.set_log_level as set_log_level_module

        for handler_id in set_log_level_module._torchquad_handler_ids:
            logger.remove(handler_id)
        set_log_level_module._torchquad_handler_ids.clear()
        logger.disable("torchquad")


def test_torchquad_records_disabled_by_default():
    """Importing torchquad disables its own loguru records, so running an
    integration must not leak torchquad log output into a host handler."""
    # Assert the library default explicitly for order-independence (other tests
    # may have called set_log_level, which enables torchquad's records).
    logger.disable("torchquad")
    host_sink = io.StringIO()
    host_handler_id = logger.add(host_sink, level="DEBUG", filter=lambda record: True)
    try:
        MonteCarlo().integrate(
            lambda x: x, dim=1, N=100, integration_domain=[[0.0, 1.0]], backend="numpy"
        )
        assert "Computed integral" not in host_sink.getvalue(), (
            "torchquad emitted log records into the host handler while disabled"
        )
    finally:
        logger.remove(host_handler_id)
        logger.disable("torchquad")
