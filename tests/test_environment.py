import pytest


def test_python_version():
    import sys
    assert sys.version_info >= (3, 11)


def test_dvc_available():
    import dvc
    assert dvc.__version__ is not None