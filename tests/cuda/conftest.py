"""The CUDA engines' tests: collected only where PyTorch sees an NVIDIA GPU (DGX Spark, in NVIDIA's container)."""

import gc
import importlib.util
import os

import pytest

# GLM's engines keep the MTP head beside DFlash2 here (TF_GLM_MTP=1), so both drafters stay under test
os.environ.setdefault("TF_GLM_MTP", "1")


def _cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


collect_ignore_glob = [] if _cuda() else ["test_*.py"]


@pytest.fixture(autouse=True, scope="module")
def _give_back_gpu_memory():
    """After each module: free its tensors and torch's cached blocks, so the next engine admits as a fresh process."""

    yield
    if not _cuda():
        return
    import torch

    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def _thor_paths_off_after():
    """Jetson Thor's chunked DeltaNet prompt chain (thor_gdn) and sparse-attention prompt kernels (thor_qsa) are
    process-wide once an engine prepares them: both are off again after each test, so the next one starts from
    TensorFold's kernels (an engine built by one test, with its cache width, does not choose the next test's path)."""

    yield
    if not _cuda():
        return
    try:
        from tensorfold.cuda.kernels import thor_gdn
        from tensorfold.families.qwen4_exp.cuda import thor_qsa
    except ImportError:
        return
    thor_gdn.set_active(False)
    thor_qsa.set_active(False)
