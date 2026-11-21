"""ChronoSync: synchronize, audit, and validate messy multi-device sensor streams."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

__all__ = ["__version__"]

try:
    __version__ = _pkg_version("chronosync-mcp")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.1.0"
