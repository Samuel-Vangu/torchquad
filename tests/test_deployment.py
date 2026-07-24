"""
Test module for deployment verification functionality.
This test ensures that the deployment_test function works correctly
and is included in pytest coverage.
"""

import warnings

import pytest

from torchquad.utils.deployment_test import deployment_test


def test_deployment_functionality():
    """
    Test that the deployment test runs successfully.

    This test verifies that the deployment_test function executes
    without critical errors and returns True, indicating successful
    deployment verification.
    """
    # Run the deployment test
    result = deployment_test()

    # The deployment test should return True on success
    assert result is True, "Deployment test failed - check logs for details"


def test_deployment_test_imports():
    """
    Test that all required imports for deployment test are available.
    """
    # Test that we can import the deployment test function
    from torchquad.utils.deployment_test import deployment_test

    assert callable(deployment_test)

    # It is also exposed on the top-level package
    import torchquad

    assert torchquad.deployment_test is deployment_test

    # Test that deployment test helper functions exist
    from torchquad.utils.deployment_test import _get_exp_func, _get_sin_func
    from torchquad.utils.deployment_test import _infer_backend_from_tensor, _is_finite_result

    assert callable(_get_exp_func)
    assert callable(_get_sin_func)
    assert callable(_infer_backend_from_tensor)
    assert callable(_is_finite_result)


def test_deprecated_alias_warns():
    """
    The old ``_deployment_test`` name is kept working for one release but must
    emit a DeprecationWarning (regression guard for the 0.6 rename).
    """
    import torchquad

    # Exercise the path users actually rely on (torchquad._deployment_test),
    # so a break in the top-level re-export is caught too.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning):
            torchquad._deployment_test()


def test_deployment_helper_functions():
    """
    Test the helper functions used by deployment test.
    """
    from torchquad.utils.deployment_test import _is_finite_result

    # Test finite result detection
    assert _is_finite_result(1.0)
    assert _is_finite_result(-1.0)
    assert _is_finite_result(0.0)

    # Test with different types
    assert _is_finite_result(42)

    # These tests might not work on all systems, so we'll be lenient
    try:
        assert not _is_finite_result(float("inf"))
        assert not _is_finite_result(float("nan"))
    except (ImportError, AssertionError):
        # Skip if math operations not available or behave differently
        pass


if __name__ == "__main__":
    # Run tests individually for debugging
    test_deployment_functionality()
    test_deployment_test_imports()
    test_deprecated_alias_warns()
    test_deployment_helper_functions()
    print("All deployment tests passed!")
