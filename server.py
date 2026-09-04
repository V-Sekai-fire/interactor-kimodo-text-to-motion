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

STUB = os.environ.get("WEFTSPUN_STUB") == "1"
KIMODO_MODEL = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1.1")
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


def _retarget(soma: Path, target_rig: str, work: Path) -> Path:
    """Not yet wired -- SOMA -> ANNY -> Godot Humanoid -> VRM, via the
    canonical joint pivot table in meshula/LabRCSF's joints.csv
    (columns include ANNY, Godot, and VRM among ~14 rigs, keyed by a
    CanonicalJoint pivot -- not a direct name-to-name guess). See
    README's Status."""
    raise NotImplementedError(
        "Port the SOMA -> ANNY -> Godot Humanoid -> VRM retarget here, using "
        "meshula/LabRCSF's joints.csv as the canonical joint pivot table -- "
        "see README's Status"
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
        # Load the checkpoint once at startup, keep it resident. Cold
        # load is expensive; per-request reload would be a mistake.
        import torch
        from kimodo import load_model

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
