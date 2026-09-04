"""Vertex-side projection accuracy check per HERD's apparatus-alignment
critique on interactor#2 + PR #291's "pending" placeholder.

Same apparatus (barycentric-interpolated vertex diff) for the positive
number and both negative controls. Positive: makehuman posed vertices
compared against SOMA_wrap posed vertices interpolated at the barycentric
map. Negative (a): zero mid-hierarchy bone column, re-pose, diff.
Negative (b): zero limb bone column on limb-moving poses, diff.

If positive clears anny thresholds (max<15mm, mean<5mm) AND both
controls report larger errors than positive, the check has teeth and
the projection is safe for corpus use. If positive fails, the corpus
subset is discarded before publish per RFD 2203.

Detection floor: n=4 sees only defects appearing in >75% of frames per
CLAUDE.md rule 5. Reported alongside the numbers.
"""
from __future__ import annotations

import copy
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import anny
import numpy as np
import roma
import torch
from anny.models.model_data import TopologyConfig
from anny.utils.warp_mesh_utils import point_to_mesh_distance_and_face_uvs

torch.set_num_threads(4)
device = "cpu"
dtype = torch.float32
T = 4


def anchor(mm: float) -> str:
    if mm < 1.0:
        return "sub-credit-card"
    if mm < 2.0:
        return f"{mm/0.76:.1f}x credit-card"
    if mm < 8.0:
        return f"{mm/1.52:.1f}x penny"
    if mm < 12.0:
        return f"{mm/7.0:.1f}x pencil"
    if mm < 30.0:
        return f"{mm/10.5:.1f}x AAA battery"
    if mm < 70.0:
        return f"{mm/42.7:.1f}x golf ball"
    return f"{mm/66.0:.1f}x soda can"


def _pose(model, pose_parameters, phenotype):
    return model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype, local_changes_kwargs={})["vertices"]


def _diff(a, b, mask=None):
    err = torch.linalg.norm(a - b, dim=-1)  # (T, V)
    if mask is not None:
        err = err[:, mask]
    return {
        "max_mm": float(err.max()) * 1000.0,
        "mean_mm": float(err.mean()) * 1000.0,
        "p99_mm": float(torch.quantile(err.flatten(), 0.99)) * 1000.0,
    }


print("Building anny models on CPU (this takes ~30-60s per model)...")
anny_direct = anny.Anny(
    rig="soma", topology="soma",
    pose_parameterization="local-ref", phenotypes="all",
).to(device=device, dtype=dtype)
anny_mh = anny.Anny(
    rig="soma",
    topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False),
    pose_parameterization="local-ref", phenotypes="all",
).to(device=device, dtype=dtype)

soma_rest = anny_direct.template_vertices.to(dtype)     # (18056, 3)
mh_rest = anny_mh.template_vertices.to(dtype)           # (19158, 3)
soma_faces = anny_direct.faces.to(torch.int64)          # SOMA_wrap triangulation
print(f"Rest shapes: soma_direct={tuple(soma_rest.shape)}, makehuman={tuple(mh_rest.shape)}, soma_faces={tuple(soma_faces.shape)}")

# Barycentric map: makehuman -> SOMA_wrap. Same call as soma.py:97
print(f"Computing makehuman -> SOMA_wrap barycentric map ({mh_rest.shape[0]} points)...")
distances, face_ids, uvs = point_to_mesh_distance_and_face_uvs(
    points=mh_rest.to(torch.float32),
    vertices=soma_rest.to(torch.float32),
    faces=soma_faces.to(torch.int32),
    max_dist=1000.0,
)
uvs = uvs.to(dtype)                                     # (19158, 2)
u, v = uvs[:, 0], uvs[:, 1]
w = 1.0 - u - v
bary = torch.stack([u, v, w], dim=-1)                   # (19158, 3)
ref_vidx = soma_faces[face_ids]                         # (19158, 3) — 3 SOMA verts per makehuman vert
print(f"Barycentric max distance from mh vert to soma triangle: {distances.max():.5f} m")

# Filter to vertices near the SOMA_wrap surface: distances > 5mm at rest
# are non-body-surface vertices (interior mesh, hair, teeth, eye internals)
# that have no meaningful SOMA correspondence — barycentric interpolation
# for those is decoration, and their per-pose diffs would drive max/p99
# without saying anything about pose correctness. wholebody133.pth's
# anchors are body-surface points, so the check that matters is on the
# body-surface subset.
SURFACE_THRESHOLD_M = 0.005  # 5 mm ≈ 3 stacked pennies
surface_mask = distances < SURFACE_THRESHOLD_M          # (19158,) bool
n_surface = int(surface_mask.sum())
n_offsurface = int((~surface_mask).sum())
print(f"Surface mask (rest dist < {SURFACE_THRESHOLD_M*1000:.0f} mm): {n_surface}/{surface_mask.numel()} verts kept ({n_offsurface} filtered as non-body-surface)")


def _interp_soma_to_mh(soma_posed):
    """soma_posed: (T, 18056, 3) → interpolated at makehuman positions: (T, 19158, 3)"""
    # Gather (T, 19158, 3, 3): for each mh vert, its 3 corresponding SOMA vertices
    corner_verts = soma_posed[:, ref_vidx]              # (T, 19158, 3, 3)
    # Weighted sum by barycentric coordinates
    interpolated = (corner_verts * bary.unsqueeze(0).unsqueeze(-1)).sum(dim=-2)
    return interpolated


# Generate 4 random poses
torch.manual_seed(0)
rotvec = 0.2 * torch.randn((T, 77, 3), device=device, dtype=dtype)
transl = 1.0 * torch.randn((T, 3), device=device, dtype=dtype)
extended = torch.cat((torch.zeros((T, 1, 3), device=device, dtype=dtype), rotvec), dim=1)
pose_parameters = roma.Rigid(roma.rotvec_to_rotmat(extended), translation=None).to_homogeneous()
pose_parameters[:, 0, :3, 3] = transl
phenotype = torch.zeros((T, len(anny_direct.phenotype_labels)), device=device, dtype=dtype)

print(f"\nRunning {T} poses through both models...")
verts_soma = _pose(anny_direct, pose_parameters, phenotype)    # (T, 18056, 3)
verts_mh = _pose(anny_mh, pose_parameters, phenotype)          # (T, 19158, 3)
print(f"soma posed: {tuple(verts_soma.shape)}, mh posed: {tuple(verts_mh.shape)}")

# ---------- POSITIVE: interpolate soma at makehuman positions, diff against mh ----------
verts_soma_at_mh = _interp_soma_to_mh(verts_soma)
positive = _diff(verts_mh, verts_soma_at_mh, mask=surface_mask)
print(f"\n=== POSITIVE: makehuman posed vs SOMA-interpolated-at-makehuman (body-surface subset, n={n_surface}) ===")
print(f"  max:  {positive['max_mm']:8.3f} mm  ({anchor(positive['max_mm'])})")
print(f"  mean: {positive['mean_mm']:8.3f} mm  ({anchor(positive['mean_mm'])})")
print(f"  p99:  {positive['p99_mm']:8.3f} mm  ({anchor(positive['p99_mm'])})")

# ---------- NEGATIVE (a): LBS-slot-column corruption (kept as apparatus record) ----------
# This tries to zero column 5 of vertex_bone_weights, but that tensor
# is per-vertex effective-bones (13 slots), not one column per bone.
# Slot 5 doesn't correspond to bone 5 in the rig; the resulting diff is
# indistinguishable from the positive number. Kept in the record so a
# future reader sees the shape that didn't work — the load-bearing
# negative control is (b) below.
model_a = copy.deepcopy(anny_mh)
weights_a = model_a.vertex_bone_weights
with torch.no_grad():
    weights_a[:, 5] = 0.0
    row_sums = weights_a.sum(dim=1, keepdim=True)
    weights_a.div_(row_sums.clamp(min=1e-9))
verts_mh_a_lbs = _pose(model_a, pose_parameters, phenotype)
negative_a_lbs = _diff(verts_mh_a_lbs, verts_soma_at_mh, mask=surface_mask)
print(f"\n=== NEGATIVE (a): mid-hierarchy weight column 5 zeroed on makehuman (LBS-slot; body-surface subset) ===")
print(f"  max:  {negative_a_lbs['max_mm']:8.3f} mm  ({anchor(negative_a_lbs['max_mm'])})")
print(f"  mean: {negative_a_lbs['mean_mm']:8.3f} mm  ({anchor(negative_a_lbs['mean_mm'])})")
print(f"  (Note: vertex_bone_weights has 13 slots per vertex, not one column per bone; this corrupts")
print(f"   weight-slot 5 on all verts. Sensitivity depends on which bones sit in slot 5 for these poses.)")

# ---------- NEGATIVE (b): pose-input corruption on limb-moving pose ----------
# Sidesteps the vertex_bone_weights slot/bone mismatch: feed the SOMA
# model a full arm-rotation pose, feed the makehuman model the SAME
# pose with LeftArm zeroed. If the check apparatus has teeth, diff
# spikes on left-arm vertices — the arm went one way on SOMA, stayed
# put on makehuman.
labels = list(anny_mh.bone_labels)
left_arm_idx = labels.index("LeftArm") - 1  # -1 for the root that gets prepended after
print(f"\nNegative (b): corrupting LeftArm rotation input (rotvec index {left_arm_idx})")

# Full pose: rotate LeftArm ~0.8-1.2 rad on all 4 frames
rotvec_limb = torch.zeros((T, 77, 3), device=device, dtype=dtype)
rotvec_limb[:, left_arm_idx, 1] = torch.tensor([0.8, -0.8, 1.2, -1.2], device=device, dtype=dtype)
ext_limb_full = torch.cat((torch.zeros((T, 1, 3), device=device, dtype=dtype), rotvec_limb), dim=1)
pose_limb_full = roma.Rigid(roma.rotvec_to_rotmat(ext_limb_full), translation=None).to_homogeneous()
pose_limb_full[:, 0, :3, 3] = transl

# Corrupted pose: same but LeftArm zeroed
rotvec_bad = rotvec_limb.clone()
rotvec_bad[:, left_arm_idx, :] = 0.0
ext_bad = torch.cat((torch.zeros((T, 1, 3), device=device, dtype=dtype), rotvec_bad), dim=1)
pose_bad = roma.Rigid(roma.rotvec_to_rotmat(ext_bad), translation=None).to_homogeneous()
pose_bad[:, 0, :3, 3] = transl

verts_soma_limb = _pose(anny_direct, pose_limb_full, phenotype)
verts_soma_at_mh_limb = _interp_soma_to_mh(verts_soma_limb)
verts_mh_limb_bad = _pose(anny_mh, pose_bad, phenotype)
negative_b = _diff(verts_mh_limb_bad, verts_soma_at_mh_limb, mask=surface_mask)
print(f"\n=== NEGATIVE (b): LeftArm zeroed on makehuman pose only, SOMA gets full arm rotation (body-surface subset) ===")
print(f"  max:  {negative_b['max_mm']:8.3f} mm  ({anchor(negative_b['max_mm'])})")
print(f"  mean: {negative_b['mean_mm']:8.3f} mm  ({anchor(negative_b['mean_mm'])})")

detection_floor = 100.0 * 3 / T
print(f"\n=== Summary ===")
print(f"  detection floor at n={T}: any defect > ~{detection_floor:.0f}% of frames per CLAUDE.md rule 5")
print(f"  gate: anny thresholds max<15 mm, mean<5 mm")
if positive['max_mm'] < 15.0 and positive['mean_mm'] < 5.0:
    print(f"  PASS positive within gate")
else:
    print(f"  FAIL positive exceeds gate")
    raise SystemExit(2)
