# srcli

A command line super-resolution tool for upscaling photographs.

`srcli` takes an image and produces a larger one, using a deep neural network to
synthesise the extra detail rather than stretching the pixels. It is built around
Real-ESRGAN, which is trained to reverse the kind of degradation real photographs
actually suffer from: sensor noise, JPEG blocking and lens blur. That makes it a
better fit for everyday images than the older bicubic-trained models, which tend
to preserve the very artifacts you were hoping to remove.

The tool is also importable as a library. The command line is a thin wrapper, so
the same pipeline can be driven from a script.

## Features

- **Real-world degradation handling.** The default model is trained on simulated
  real degradations, so it cleans up noise and compression while sharpening.
- **Arbitrary output sizes.** Ask for a factor (`--scale 4`) or an exact size
  (`--size 3840x2160`). A request that does not land on a model's native factor is
  met by running the model and then resampling to the exact target.
- **Model chaining.** When no single model reaches the requested factor, several
  are composed automatically, for example two x4 passes to cover x16.
- **Tiled inference with feathered blending.** Large images are processed in
  overlapping tiles and blended with a crossfade window, so the seams between
  tiles are not visible.
- **CPU, CUDA and Apple MPS.** The compute device is detected automatically and
  can be overridden.
- **Weights fetched on first use.** Nothing heavy ships in the package; model
  files are downloaded to a shared cache and reused afterwards.
- **An explicit model registry.** Every model names its own architecture, weight
  source and native factor, so the network is always constructed deliberately
  rather than guessed from a checkpoint.

## Installation

From the repository root:

```bash
pip install -e .
```

This gives you the `sr` command. It pulls in PyTorch and the Hugging Face Hub
client, which are needed to build a network and fetch its weights. If you want a
CUDA build of PyTorch, install it first from the appropriate index, then install
`srcli`; pip will keep the version you already have.

For development, install the test and lint tools as well:

```bash
pip install -e ".[dev]"
```

If you would rather not install anything, run it as a module with the repository
root on the import path:

```bash
PYTHONPATH=. python3 -m srcli.cli --help
```

## Quick start

Upscale an image by four times:

```bash
sr photo.jpg -o photo_4x.png --scale 4
```

Produce an exact resolution:

```bash
sr photo.jpg -o photo_4k.png --size 3840x2160
```

List the models the registry knows about:

```bash
sr models
```

## Use as a library

The same upscaling logic is available from Python. `upscale` takes an input and
output path and the same options as the command line, and returns the output
path:

```python
from srcli.pipeline import upscale

# Fixed factor, the model chosen automatically.
upscale("photo.jpg", "photo_4x.png", scale=4)

# An exact target resolution.
upscale("photo.jpg", "photo_4k.png", target_size=(3840, 2160))

# A named model and an explicit device, skipping automatic selection.
upscale("photo.jpg", "out.png", model="hat-s-x4", device="cuda", fp16=True)
```

`scale` and `target_size` are mutually exclusive, exactly as on the command line,
and every other argument defaults to the value the CLI would have derived. To
pick a model by capability rather than by name, pass `tags`:

```python
upscale("photo.jpg", "out.png", scale=4, tags=["transformer"])
```

For a custom pipeline the lower level pieces are importable too. `load_image`
returns a three channel tensor and `save_image` writes one back out; `load_by_name`
returns a runnable model whose `upscale` expects a tensor already padded to the
model's `alignment`:

```python
from srcli.io import load_image, save_image
from srcli.models.loader import load_by_name

model = load_by_name("hat-s-x4", device="cuda")
print(model.scale, model.alignment)
```

Most callers want `srcli.pipeline.upscale` above, which handles the padding, the
`no_grad` context and the final resample for you.

## Usage

```
sr upscale [-h] -o OUTPUT [--scale SCALE] [--size SIZE] [--model MODEL]
           [--tag TAGS] [--device DEVICE] [--fp16] [--tile TILE]
           input
```

The `upscale` command is the default, so `sr photo.jpg -o out.png` and
`sr upscale photo.jpg -o out.png` are the same.

### Options

| Option | Description |
|---|---|
| `input` | Path to the image to upscale. |
| `-o`, `--output` | Path to write the result to. The format is taken from the extension. |
| `--scale` | Upscale factor, for example `4` for four times larger. |
| `--size` | Exact output size as `WIDTHxHEIGHT`, for example `3840x2160`. |
| `--model` | A model name, or `auto` to pick one for the requested factor. Defaults to `auto`. |
| `--tag` | Prefer models with this tag when `--model` is `auto`. Repeatable, for example `--tag transformer` reaches the HAT models. |
| `--device` | Compute device: `cpu`, `cuda` or `mps`. Detected automatically if omitted. |
| `--fp16` | Use half precision on CUDA, to save memory and time. |
| `--tile` | Input pixel budget before tiling kicks in. Lower values use less memory. |

`--scale` and `--size` are mutually exclusive. If neither is given, the model's
own factor is used.

### Examples

Upscale on a specific device:

```bash
sr photo.jpg -o out.png --scale 4 --device cuda --fp16
```

Keep memory low for a large image by shrinking the tile budget:

```bash
sr huge.jpg -o huge_4x.png --scale 4 --tile 262144
```

Ask for a factor no single model provides; `auto` chains two x4 passes to cover
x16 and then resamples to x10:

```bash
sr photo.jpg -o photo_10x.png --scale 10
```

Run a specific model by name:

```bash
sr photo.jpg -o out.png --model hat-s-x4
```

Let `auto` select within a capability rather than a fixed model; the `transformer`
tag picks the cheaper small HAT and skips the large one:

```bash
sr photo.jpg -o out.png --scale 4 --tag transformer
```

## How it works

The pipeline runs in a fixed order, which keeps the ordering mistakes in one
place:

1. **Load** the image as a three channel float tensor in the `0..1` range.
2. **Resolve the plan.** The requested factor or size becomes an explicit target,
   and the alignment the network needs is computed.
3. **Pad** the image to that alignment using reflection padding, remembering the
   region that corresponds to real pixels.
4. **Infer**, either in a single pass or tile by tile when the image is large.
5. **Crop** away the padding halo so what remains is exactly the scaled image.
6. **Resample** to the exact requested size, if the model factor did not land on
   it already.

### A note on arbitrary sizes

A model has a fixed native factor: a x4 model cannot natively produce x3. To hit
an arbitrary size, `srcli` runs the nearest model that covers the request and
then resamples to the exact target. Overshooting and shrinking down looks better
than trying to synthesise a fractional factor directly.

For very large factors, several models are chained, and the intermediate result
is larger than the final one. A x10 request is served by two x4 passes, so the
x16 intermediate is what has to fit in memory, not the x10 output.

### A note on tiling

Large images are split into overlapping tiles so a single pass never has to hold
the whole thing at once. The tiles are blended with a weighted window that fades
only on the sides that meet another tile, so the border pixels are not dimmed and
the overlaps sum to one. Without that crossfade, each tile's slightly different
edge output would show up as a grid.

## Models

The registry describes each model completely: how to build the network, where its
weights come from, and what factor it provides. Run `sr models` to list them.

| Name | Scale | Source | Notes |
|---|---|---|---|
| `realesrgan-x4plus` | x4 | Hugging Face | General purpose photo model, the default for most requests. |
| `realesrgan-x2plus` | x2 | Direct URL | Same family, native x2, useful when a gentler factor is enough. |
| `hat-s-x2` / `hat-s-x3` / `hat-s-x4` | x2/x3/x4 | Google Drive | Hybrid Attention Transformer, small width. Higher quality on textures, heavier to run. |
| `hat-l-x2` / `hat-l-x3` / `hat-l-x4` | x2/x3/x4 | Google Drive | Same architecture at double the depth. Best quality, slowest. |

The HAT models are attention based and require the input width and height to be a
multiple of sixteen, which the pipeline pads to automatically. They are not part
of the automatic selection pool (they do not carry the `general` tag), so they are
chosen by name with `--model hat-s-x4`, or by opting in with `--tag transformer`.
When several models offer the same factor the lower ranked one wins, so the tag
prefers the cheaper small width over the large one.

Weights are cached on first use. The Hugging Face Hub models use the Hub's own
cache and authentication; direct URL and Google Drive models are downloaded into a
shared cache directory and verified against a pinned checksum. The Google Drive
source needs the optional `gdown` dependency, installed with
`pip install 'srcli[gdrive]'`.

### Adding a model

Adding a model that shares an existing architecture is one registry entry.
Adding a new architecture is a builder function plus an entry. Nothing in the
pipeline, the tiling code or the command line needs to change.

Point the entry at a builder, a weight source and the factor those weights
provide:

```python
# srcli/models/registry.py
"my-model-x4":
    ModelEntry(
        name="my-model-x4",
        scale=4,
        build=builders.build_my_model_x4,
        weights=HubFile(
            repo_id="owner/repo",
            filename="my-model-x4.pth",
        ),
        alignment=16,
    ),
```

A model with a new architecture also needs a builder that returns the network:

```python
# srcli/models/builders.py
def build_my_model_x4():
  return MyModel(upscale=4)
```

Once the entry is registered, `sr models` lists it, `--model my-model-x4` runs it,
and any tag it carries makes it eligible for `--tag`.

## Performance

Super-resolution is compute heavy, and the cost grows with the number of output
pixels. A GPU is strongly recommended for anything beyond small images; on a CPU
a four times upscale of a modest photo takes minutes, and a chained x10 run can
take tens of minutes.

A few practical notes:

- **Prefer the smallest factor that meets your need.** A single `--scale 4` pass
  is far cheaper, and usually looks better per pixel, than a chained run.
- **Cap the thread count on CPU.** Oversubscribing threads can make things
  several times slower. For example, `OMP_NUM_THREADS=8 sr photo.jpg -o out.png
  --scale 4`.
- **Lower `--tile` if you run out of memory.** Tiling reduces per-tile compute but
  not the size of the final canvas, so budget memory for the output too.

## Development

Run the tests:

```bash
python3 -m pytest tests/
```

Lint the code:

```bash
ruff check srcli tests
```

Format the code:

```bash
yapf -i -r --style .style.yapf srcli tests
```

The geometry, tiling, selection and loader modules are covered by unit tests, and
the pipeline is exercised end to end with a stand-in model so the tests run
without downloading weights.

## License

The `srcli` source is provided as is. Note that model weights carry their own
licenses, which may differ from the code and may restrict commercial use. Check
the terms of any model before using it in a commercial product.
