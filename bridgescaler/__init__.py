from importlib.metadata import version, PackageNotFoundError

from packaging.version import Version


# 1. PyTorch Checks
REQUIRED_TORCH_VERSION = Version("2.0.0")

def get_torch_status() -> tuple[bool, Version | None]:
    try:
        return True, Version(version("torch"))
    except PackageNotFoundError:
        return False, None

TORCH_AVAILABLE, TORCH_VERSION = get_torch_status()

def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required but not installed")
    if TORCH_VERSION < REQUIRED_TORCH_VERSION:
        raise RuntimeError(
            f"PyTorch >= {REQUIRED_TORCH_VERSION} required; found {TORCH_VERSION}"
        )

# 2. Base Imports
from .backend import save_scaler, load_scaler, print_scaler, read_scaler
from .group import GroupStandardScaler, GroupRobustScaler, GroupMinMaxScaler
from .deep import DeepStandardScaler, DeepMinMaxScaler, DeepQuantileTransformer
from .distributed import (DStandardScaler, DMinMaxScaler, DQuantileScaler)

# 3. Conditional Torch Imports
if TORCH_AVAILABLE:
    try: # Ensure that no errors are raised if PyTorch is installed but does not meet the required version.
        from .distributed_tensor import (
            DStandardScalerTensor,
            DMinMaxScalerTensor,
        )
    except:
        pass

# 4. Define Public API
__all__ = [
    # Utilities
    "save_scaler", "load_scaler", "print_scaler", "read_scaler",
    "TORCH_AVAILABLE",
    # Scalers
    "GroupStandardScaler", "GroupRobustScaler", "GroupMinMaxScaler",
    "DeepStandardScaler", "DeepMinMaxScaler", "DeepQuantileTransformer",
    "DStandardScaler", "DMinMaxScaler", "DQuantileScaler",
]
