# interactor-kimodo-text-to-motion

Model image for `kimodo_text_to_motion`, per
[weftspun's RFD 0036](https://github.com/weftspun/request-for-discussion/tree/main/0036-packaging-convention)
packaging convention. Facts from
[RFD 0045](https://github.com/weftspun/request-for-discussion/tree/main/0045-kimodo-text-to-motion).

## Model

| Property | Value |
|---|---|
| Upstream | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) (NVIDIA), kinematic motion diffusion |
| License | Code: Apache-2.0. Weights: **per-checkpoint** — see below, this was reviewed and resolved, not left pending |
| Parameters | 0.3 B, estimated |
| bf16 | 0.6 GB — the ship format (no Q4_K_M at this size) |

### License, resolved

RFD 0045 didn't record a weight license. Checked directly against `nv-tlabs/kimodo`'s own README:
code is Apache-2.0, but **checkpoints are licensed separately per variant** —

- `Kimodo-SOMA` and `Kimodo-G1`: **NVIDIA Open Model License** (commercial-friendly)
- `Kimodo-SMPLX`: **NVIDIA R&D Model License** (research-only, more restrictive)

This image ships `Kimodo-SOMA` — it matches what this interface actually returns by default (SOMA
tracks; VRM only if `target_rig` is given), and it's the commercially-clear variant. **Do not swap
in the SMPLX checkpoint** without re-checking RFD 0028's commercial-use license gate first — that
variant would fail it.

## Interface

`POST /predict`:

| Input | Type | Default | Note |
|---|---|---|---|
| `prompt` | str | required | The motion, as a sentence |
| `duration_seconds` | float | 4.0 | 0.5–30.0 |
| `fps` | int | 30 | 12–120 |
| `target_rig` | Path/URL/base64 VRM | none | Optional. Without it, SOMA only, no retarget |
| `seed` | int | -1 | |

Returns `{soma, vrm, valid, validation_detail, seed, stub}`. `soma` and `vrm` are kept as separate
fields on purpose (RFD 0045): a single merged output would make the model look wrong when it's
actually the retarget step that failed. `vrm` is `null` unless `target_rig` was given.

### The retarget path: SOMA → ANNY → Godot Humanoid → VRM

The retarget doesn't map SOMA joint names straight to VRM's — it goes through a canonical pivot.
[meshula/LabRCSF](https://github.com/meshula/LabRCSF) documents a Reference Canonical Skeletal
Framework: `joints.csv` maps ~14 skeletal rig standards (including ANNY, Godot, VRM, SMPL, Mixamo,
HAnim, and others) to a shared `CanonicalJoint` column, e.g. `Hips`, `Spine`, `Chest`, `Neck`,
`Head` each carry their per-rig name across every format in the same row. Going through this pivot
(SOMA output joints → ANNY's naming → Godot Humanoid → VRM) means the retarget uses a
well-documented, cross-checked mapping table instead of an inline name-to-name guess — and RFD
0046's "joint order trap" (VRM names joints, USD/SOMA don't, in the same order) is exactly the kind
of problem `joints.csv` exists to solve. Not yet wired — see Status.

## The validation gate (RFD 0007)

RFD 0007's motion validation runs inside this image, not downstream. A motion that leaves the
floor or inverts a knee must fail here — `valid`/`validation_detail` in the response tell a caller
whether the motion is usable before it ever loads into a viewport.

## Build

```sh
docker build --target contract -t interactor-kimodo-text-to-motion:contract .
docker run --rm -p 8000:8000 interactor-kimodo-text-to-motion:contract
curl -X POST localhost:8000/predict -d @test_input.json -H 'Content-Type: application/json'
```

## Status

**Scaffolded from the RFD, not yet built or run.** `_run_upstream()`, `_retarget()`, and
`_validate_motion()` all raise `NotImplementedError` outside stub mode — Kimodo's real sampler
call, the SOMA → ANNY → Godot Humanoid → VRM retarget (via `meshula/LabRCSF`'s `joints.csv`), and
RFD 0007's floor-contact/joint-limit checks are not yet ported from the upstream repos.
`KIMODO_REPO` in the Dockerfile (`nv-tlabs/kimodo-soma`) is a best guess at the HF repo id for the
SOMA checkpoint specifically — confirm the exact id/filename before trusting the worker stage's
weight fetch.
