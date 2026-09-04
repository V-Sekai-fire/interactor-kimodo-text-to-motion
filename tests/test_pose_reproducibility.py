"""Deterministic-reproducibility gate + projection-path verification for
the SOMA -> ANNY pose step per RFD 2203 review.

Two gates:

1. Same-seed same-prompt reproduces posed ANNY vertices to 1e-6. The
   sampler is deterministic; the anny forward is deterministic; the
   composition must be too. If a run drifts, either seeding is not
   reaching the sampler or the anny model is picking up a non-seeded
   randomness (a random phenotype under 'all', a per-request torch
   default generator, etc.).

2. Projection-path accuracy check per RFD 2203. anny/test/test_soma.py
   verified the direct topology='soma' path against upstream SOMALayer
   at ~7mm max / ~0.6mm mean. This function uses
   topology=TopologyConfig(base_mesh='makehuman'), which routes through
   apply_procrustes_retopology -- the projection has no accuracy check
   in anny's own tests. This gate compares the makehuman-projected
   posed vertices against the direct topology='soma' path at shared
   joint positions and gates at anny's own thresholds (max<15mm,
   mean<5mm). A regression beyond the pencil is a spec finding, not a
   silent cost.

Runs the real sampler + real anny; needs CUDA + the kimodo package +
anny. Skipped under WEFTSPUN_STUB=1.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("WEFTSPUN_STUB") == "1":
    pytest.skip("real sampler + anny required", allow_module_level=True)

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("torch required", allow_module_level=True)


PROMPT = "A person walks forward."
DURATION = 2.0
FPS = 30
SEED = 20260904


@pytest.fixture(scope="module")
def loaded():
    from server import load

    load()


def _sample_and_pose(prompt: str, seed: int) -> np.ndarray:
    from server import _pose_anny_from_soma, _run_upstream

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        soma_path = _run_upstream(
            {"prompt": prompt, "duration_seconds": DURATION, "fps": FPS, "seed": seed},
            work,
        )
        anny_path = _pose_anny_from_soma(soma_path, work)
        data = np.load(anny_path, allow_pickle=False)
        return data["vertices"]


def test_same_seed_reproduces_posed_vertices(loaded):
    a = _sample_and_pose(PROMPT, SEED)
    b = _sample_and_pose(PROMPT, SEED)
    assert a.shape == b.shape
    # 19158 vertices per frame at the makehuman topology
    assert a.shape[-2] == 19158, f"expected 19158-vertex output, got {a.shape}"
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_different_seed_differs(loaded):
    """Negative control: seed plumbing might silently no-op, in which
    case the positive test above passes even on a broken sampler."""
    a = _sample_and_pose(PROMPT, SEED)
    b = _sample_and_pose(PROMPT, SEED + 1)
    assert a.shape == b.shape
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_makehuman_projection_accuracy(loaded):
    """Verify the projection through apply_procrustes_retopology stays
    within anny's own thresholds (max<15mm, mean<5mm) against the
    direct topology='soma' path at the shared 6th spinal joint (a
    joint present in both topologies via the anny bone hierarchy).

    A tighter bound is better, and if projection lands within a pencil
    of the direct path the RFD 2203 verification hook stands with no
    further comment. Beyond a pencil is stated cost per DETAILS.
    """
    import anny
    from anny.models.model_data import TopologyConfig
    import roma

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    anny_soma_direct = anny.Anny(
        rig="soma", topology="soma", pose_parameterization="local-ref", phenotypes="all",
    ).to(device=device, dtype=dtype)
    anny_soma_makehuman = anny.Anny(
        rig="soma",
        topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False),
        pose_parameterization="local-ref",
        phenotypes="all",
    ).to(device=device, dtype=dtype)

    torch.manual_seed(0)
    T = 4
    rotvec = 0.2 * torch.randn((T, 77, 3), device=device, dtype=dtype)
    transl = 1.0 * torch.randn((T, 3), device=device, dtype=dtype)
    extended = torch.cat((torch.zeros((T, 1, 3), device=device, dtype=dtype), rotvec), dim=1)
    pose_parameters = roma.Rigid(roma.rotvec_to_rotmat(extended), translation=None).to_homogeneous()
    pose_parameters[:, 0, :3, 3] = transl
    phenotype = torch.zeros((T, len(anny_soma_direct.phenotype_labels)), device=device, dtype=dtype)

    out_direct = anny_soma_direct(pose_parameters=pose_parameters, phenotype_kwargs=phenotype, local_changes_kwargs={})
    out_mh = anny_soma_makehuman(pose_parameters=pose_parameters, phenotype_kwargs=phenotype, local_changes_kwargs={})

    # Joint positions are what compare cleanly across topologies (vertices
    # don't correspond 1:1 between meshes). Both models expose posed
    # joint positions under 'joints' or similar; check the shared bones.
    j_direct = out_direct.get("joints")
    j_mh = out_mh.get("joints")
    assert j_direct is not None and j_mh is not None, "anny output missing 'joints' key"
    assert j_direct.shape == j_mh.shape, f"joint counts differ: {j_direct.shape} vs {j_mh.shape}"
    err = torch.linalg.norm(j_direct - j_mh, dim=-1)
    max_mm = float(err.max()) * 1000.0
    mean_mm = float(err.mean()) * 1000.0
    print(f"projection joint error: max={max_mm:.2f} mm  mean={mean_mm:.2f} mm")
    # anny's own thresholds from test_soma.py:292-293
    assert max_mm < 15.0, f"projection joint max {max_mm:.2f} mm exceeds 15 mm gate"
    assert mean_mm < 5.0, f"projection joint mean {mean_mm:.2f} mm exceeds 5 mm gate"
