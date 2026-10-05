# interactor-kimodo-text-to-motion

An HTTP model server that turns a sentence into body motion on the SOMA skeleton and poses the ANNY body with it.

## What it is for

It samples a text-to-motion diffusion model and returns the motion on the model's own skeleton next to the ANNY body it poses, for the constructed motion corpus. A retarget to a supplied rig is a separate result, so a failed retarget is not mistaken for a failed model. [RFD 1045](https://github.com/V-Sekai-fire/manuals-weftspun/tree/main/rfd/1045-kimodo-text-to-motion) owns the interface and [RFD 1036](https://github.com/V-Sekai-fire/manuals-weftspun/tree/main/rfd/1036-packaging-convention) the packaging convention.

## Build and run

```sh
docker build --target worker .
```

The `contract` stage serves a stub of the same interface without a GPU or weights.

## Licence

This repository states no licence. The upstream model code and the checkpoint the image downloads carry their own licences.
