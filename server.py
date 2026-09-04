"""Kimodo text to motion. RFD 0045.

The model emits SOMA. The client wants VRM humanoid tracks. Those are
two things, thus this server returns two things -- a single VRM output
would make the model look wrong when the retarget is what failed.
Keeping them apart names the failure correctly.

RFD 0007's motion validation runs here. A motion that leaves the floor
or that inverts a knee must fail here, and not in the viewport.
"""

import base64
import os
import tempfile
from pathlib import Path

# CPU cap for the anny SOMA-rig posing step. VRChat interactivity on this
# box is the reason; 4 threads for anny's batched forward leaves headroom
# for the interactive GPU's driver work. Set before torch import so it
# takes effect on the first forward.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

STUB = os.environ.get("WEFTSPUN_STUB") == "1"
KIMODO_MODEL = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.1")
ANNY_MODEL_CACHE = {"anny": None}
_READY = {"loaded": False}
_MODEL = {"model": None, "resolved_name": None, "device": None}


class InputError(ValueError):
    """The request is wrong. This is the caller's fault, and not ours."""


def _validate(job_input: dict) -> dict:
    if not job_input.get("prompt"):
        raise InputError("prompt is required")
    duration = float(job_input.get("duration_seconds", 4.0))
    if not (0.5 <= duration <= 30.0):
        raise InputError("duration_seconds must be between 0.5 and 30.0")
    fps = int(job_input.get("fps", 30))
    if not (12 <= fps <= 120):
        raise InputError("fps must be between 12 and 120")
    return {
        "prompt": job_input["prompt"],
        "duration_seconds": duration,
        "fps": fps,
        "target_rig": job_input.get("target_rig"),
        "seed": int(job_input.get("seed", -1)),
    }


def _run_upstream(args: dict, work: Path) -> Path:
    """Sampler over Kimodo-SOMA-RP-v1.1 (NVIDIA Open Model License).
    Emits at the model's intrinsic fps; caller's requested fps is
    honoured for `num_frames` computation but not resampled downstream --
    the response reports the emitted fps in `model_fps` so the retarget
    stage can resample to the target rig's rate if needed."""
    import numpy as np
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.tools import seed_everything

    if _MODEL["model"] is None:
        raise RuntimeError("kimodo model not loaded; load() must run first")
    model = _MODEL["model"]

    # A single-sentence prompt is one text; the upstream CLI splits on
    # periods to support multi-prompt sequences. Preserve that so
    # callers who send "Walk forward. Then turn left." get the
    # transition sequence rather than one prompt with a period in it.
    texts = [t.strip() + "." for t in args["prompt"].split(".") if t.strip()]
    if not texts:
        raise InputError("prompt collapsed to empty after splitting on periods")

    # num_frames is per-prompt; use the same duration for each split
    # unless the caller sends a multi-duration string in a later
    # revision. Compute at model.fps because that's the sampling rate;
    # target-rig retarget resamples if it needs a different rate.
    per_prompt_frames = int(args["duration_seconds"] * model.fps)
    num_frames = [per_prompt_frames] * len(texts)

    if args["seed"] >= 0:
        seed_everything(args["seed"])

    output = model(
        texts,
        num_frames,
        constraint_lst=[],
        num_denoising_steps=100,
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=5,
        post_processing=True,
        return_numpy=True,
    )

    # `output` is a dict of arrays with leading n_samples dim; strip
    # that dim for the single-sample case so save_kimodo_npz gets the
    # per-clip shape it expects.
    single = {
        k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
        for k, v in output.items()
    }
    npz_path = work / "output.npz"
    save_kimodo_npz(str(npz_path), single)
    return npz_path


def _pose_anny_from_soma(soma_npz: Path, work: Path) -> Path:
    """v1 retarget scope per RFD 2203: SOMA -> ANNY posed vertices only.
    No Godot, no VRM. Loads Kimodo's local rotations + root positions
    from the .npz, drives an anny model wearing the SOMA rig on the
    19,158-vertex makehuman topology (matches wholebody133.pth's
    indexing), returns (T, 19158, 3) posed vertices as an .npz.

    The one-root-identity prepend: Kimodo emits 77 SOMA joint rotations;
    anny takes 78 pose parameters (root at index 0 + 77). anny/test/
    test_soma.py:242-283 shows the pattern verified against upstream
    SOMALayer at ~7mm max / ~0.6mm mean on topology=soma. This function
    uses topology=makehuman which routes through
    apply_procrustes_retopology; that projection path is
    accuracy-checked by test_pose_reproducibility.py."""
    import numpy as np
    import roma
    import torch

    torch.set_num_threads(4)

    anny_model = ANNY_MODEL_CACHE["anny"]
    if anny_model is None:
        raise RuntimeError("anny model not loaded; load() must run first")

    device = _MODEL.get("device") or "cpu"
    dtype = torch.float32
    data = np.load(str(soma_npz), allow_pickle=False)
    local_rot_mats = torch.from_numpy(np.asarray(data["local_rot_mats"])).to(device=device, dtype=dtype)
    root_positions = torch.from_numpy(np.asarray(data["root_positions"])).to(device=device, dtype=dtype)

    # local_rot_mats shape is (T, J, 3, 3). anny takes (B, 78, 4, 4)
    # homogeneous transforms with root at index 0. Kimodo's J = 77 for
    # SOMA-77 skeleton, so prepend one identity rotation for root.
    T = int(local_rot_mats.shape[0])
    J = int(local_rot_mats.shape[1])
    if J not in (77, 78):
        raise RuntimeError(f"unexpected SOMA joint count J={J}; expected 77 (root separate) or 78 (root at index 0)")

    rotvec = roma.rotmat_to_rotvec(local_rot_mats)
    if J == 77:
        root_pad = torch.zeros((T, 1, 3), device=device, dtype=dtype)
        rotvec_ext = torch.cat((root_pad, rotvec), dim=1)
    else:
        rotvec_ext = rotvec

    pose_rotmat = roma.rotvec_to_rotmat(rotvec_ext)
    pose_parameters = roma.Rigid(pose_rotmat, translation=None).to_homogeneous()
    pose_parameters[:, 0, :3, 3] = root_positions

    # Neutral phenotype + no local changes for v1: a single average body,
    # not a distribution over phenotypes. Corpus renders can vary
    # phenotype via a separate mechanism; the retarget itself doesn't
    # need to.
    phenotype_kwargs = torch.zeros((T, len(anny_model.phenotype_labels)), device=device, dtype=dtype)
    local_changes_kwargs = dict()

    output = anny_model(
        pose_parameters=pose_parameters,
        phenotype_kwargs=phenotype_kwargs,
        local_changes_kwargs=local_changes_kwargs,
    )
    vertices = output["vertices"]
    if vertices.shape[-2] != 19158:
        raise RuntimeError(
            f"anny returned vertices.shape={tuple(vertices.shape)}; expected trailing 19158 "
            f"to match wholebody133.pth indexing. Check TopologyConfig(remove_unattached_vertices=False)."
        )

    out_path = work / "anny_posed_vertices.npz"
    np.savez(str(out_path), vertices=vertices.detach().cpu().numpy(), fps=float(_MODEL["model"].fps if _MODEL["model"] else 30.0))
    return out_path


def _retarget(soma: Path, target_rig: str, work: Path) -> Path:
    """v2 deliverable: SOMA -> ANNY -> Godot Humanoid -> VRM via
    LabRCSF's joints.csv. Not in v1 scope per RFD 2203; v1 stops at
    ANNY posed vertices."""
    raise NotImplementedError(
        "VRM export deferred to v2. Use _pose_anny_from_soma() for v1's "
        "ANNY-posed-vertices output; see RFD 2203 for scope."
    )


def _validate_motion(soma: Path) -> tuple:
    """RFD 0007. Check floor contact and joint limits. A motion that
    leaves the floor or inverts a knee must fail here, before a
    caller ever loads it into a viewport."""
    if STUB:
        return True, "stub: validation not run"
    raise NotImplementedError("Port RFD 0007's validation gate here -- see README's Status")


def _encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def predict(job_input: dict) -> dict:
    args = _validate(job_input)
    work = Path(tempfile.mkdtemp())

    if STUB:
        soma = work / "stub.soma"
        soma.write_bytes(b"SOMAstub")
    else:
        soma = _run_upstream(args, work)

    valid, detail = _validate_motion(soma)

    # v1: always pose ANNY from SOMA output. target_rig gates VRM only,
    # which is v2 (still NotImplementedError). ANNY-posed-vertices is
    # what RFD 2203's corpus render consumes; no target_rig required.
    if STUB:
        anny_path = work / "stub.anny_vertices.npz"
        anny_path.write_bytes(b"ANNYvertsSTUB")
    else:
        anny_path = _pose_anny_from_soma(soma, work)

    vrm = None
    if args["target_rig"]:
        if STUB:
            vrm_path = work / "stub.vrm"
            vrm_path.write_bytes(b"glTFstub")
            vrm = _encode(vrm_path)
        else:
            vrm = _encode(_retarget(soma, args["target_rig"], work))

    return {
        "soma": _encode(soma),
        "anny_posed_vertices": _encode(anny_path),
        "vrm": vrm,
        "valid": valid,
        "validation_detail": detail,
        "seed": args["seed"],
        "stub": STUB,
        "model": None if STUB else _MODEL["resolved_name"],
        "model_fps": None if STUB or _MODEL["model"] is None else int(_MODEL["model"].fps),
    }


def load() -> None:
    if not STUB:
        # Load both models once at startup, keep resident. Cold load is
        # expensive; per-request reload would be a mistake.
        import anny
        import torch
        from anny.models.model_data import TopologyConfig
        from kimodo import load_model

        torch.set_num_threads(4)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, resolved = load_model(
            KIMODO_MODEL,
            device=device,
            default_family="Kimodo",
            return_resolved_name=True,
        )
        _MODEL["model"] = model
        _MODEL["resolved_name"] = resolved
        _MODEL["device"] = device

        # SOMA rig on the 19,158-vertex makehuman topology per RFD 2203
        # topology settlement. remove_unattached_vertices=False keeps
        # the full 19,158 count that wholebody133.pth indexes against;
        # the default True drops vertices and breaks the anchor
        # indexing.
        ANNY_MODEL_CACHE["anny"] = anny.Anny(
            rig="soma",
            topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False),
            pose_parameterization="local-ref",
            phenotypes="all",
        ).to(device=device, dtype=torch.float32)
    _READY["loaded"] = True


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="kimodo_text_to_motion", version="0.1.0")

    class PredictRequest(BaseModel):
        prompt: str
        duration_seconds: float = 4.0
        fps: int = 30
        target_rig: str | None = None
        seed: int = -1

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": _READY["loaded"], "stub": STUB}

    @app.post("/predict")
    def run(request: PredictRequest):
        try:
            return predict(request.model_dump())
        except InputError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})

    return app


if __name__ == "__main__":
    import uvicorn

    load()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
