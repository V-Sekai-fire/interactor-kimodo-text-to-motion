# interactor-kimodo-text-to-motion

An HTTP model server for turning a sentence into body motion on the SOMA skeleton and posing the ANNY body with it.

## What it is for

It samples a text-to-motion diffusion model and returns the motion on the model's own skeleton next to the ANNY body it poses, for the constructed motion corpus. [RFD 1045](https://github.com/V-Sekai-fire/manuals-weftspun/tree/main/rfd/1045-kimodo-text-to-motion) owns the interface and [RFD 1036](https://github.com/V-Sekai-fire/manuals-weftspun/tree/main/rfd/1036-packaging-convention) the packaging convention.

## Status

Only the `contract` stage answers end to end; it serves a stub of the interface without a GPU or weights. In the `worker` stage the motion check and the retarget to a supplied rig raise `NotImplementedError`, so every request fails there.

## Build and run

```sh
docker build --target contract -t kimodo-contract .
docker run --rm -p 8000:8000 kimodo-contract
```

## Licence

This repository states no licence. The upstream model code and the checkpoint the image downloads carry their own licences.
