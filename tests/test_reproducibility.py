"""Deterministic-reproducibility gate. Same seed + same prompt reproduces
the same SOMA bytes on the same box + same checkpoint.

Runs the real sampler (not the stub), so it needs CUDA + the kimodo
package installed. Skipped when `WEFTSPUN_STUB=1` because the stub
returns constant bytes and there is nothing to reproduce.

Two candidate builds run back-to-back; comparison is on the
posed_joints array (nan-safe elementwise). Byte-level comparison of the
saved .npz is not the check because npz zip metadata can differ across
runs even when the tensor contents match.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("WEFTSPUN_STUB") == "1":
    pytest.skip("real sampler required", allow_module_level=True)

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("torch required", allow_module_level=True)


PROMPT = "A person walks forward."
DURATION = 4.0
FPS = 30
SEED = 20260904


@pytest.fixture(scope="module")
def loaded():
    from server import load

    load()


def _sample(prompt: str, seed: int) -> np.ndarray:
    from server import _run_upstream

    with tempfile.TemporaryDirectory() as tmp:
        path = _run_upstream(
            {"prompt": prompt, "duration_seconds": DURATION, "fps": FPS, "seed": seed},
            Path(tmp),
        )
        data = np.load(path, allow_pickle=False)
        return data["posed_joints"]


def test_same_seed_reproduces_motion(loaded):
    a = _sample(PROMPT, SEED)
    b = _sample(PROMPT, SEED)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_different_seed_differs(loaded):
    """Negative control: a passing repro test could equally be a
    seed-ignored sampler returning the same output regardless. If two
    different seeds produce identical motion, the seed plumbing is
    broken and the positive test is decoration."""
    a = _sample(PROMPT, SEED)
    b = _sample(PROMPT, SEED + 1)
    assert a.shape == b.shape
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
