"""Perform one full-resolution GPU optimizer step in a fresh process.

The Windows CUDA build used for this project retains progressively larger
decoder workspaces across repeated research-model steps.  A fresh process per
step is therefore the honest way to keep the requested 1.8 GiB host-RSS cap
while still training the research model on the GPU.  The checkpoint contains
only BF16 model weights and is overwritten atomically by the caller.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.models import ManifestRecord, WavelengthMetadata  # noqa: E402
from dataset.parent_loader import FitsManifestParentLoader, ManifestParentLoader  # noqa: E402
from model import research_config  # noqa: E402
from synthetic import SyntheticConfig  # noqa: E402
from training import (  # noqa: E402
    AstroMambaHTrainingAdapter,
    BoundedTrainer,
    TrainingConfig,
    default_loss_fn,
    event_only_loss_fn,
    source_event_loss_fn,
    iter_paired_synthetic_training_batches,
    iter_parented_synthetic_training_batches,
)
from training.pipeline import _split_batch  # noqa: E402


def prepare_worker_batch(batch, device):
    """Stage the only active batch before training and release CPU storage."""

    return batch.to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--view", type=int, choices=(0, 1), required=True)
    parser.add_argument("--phase", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--exposure-index",
        type=int,
        default=None,
        help="optional single-exposure diagnostic; default loads the full parent sequence",
    )
    parser.add_argument("--target-loss", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--source-event-curriculum", action="store_true")
    parser.add_argument("--visits", type=int, default=1)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument(
        "--skip-dense-heatmaps",
        action="store_true",
        help="disable dense decoder activations for memory-constrained sequence training",
    )
    parser.add_argument(
        "--sequence-summary",
        action="store_true",
        help="consume the full parent and reduce it to a cap-safe temporal summary",
    )
    args = parser.parse_args()
    if args.event_only and args.source_event_curriculum:
        raise ValueError("event-only and source-event-curriculum are mutually exclusive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.visits < 1 or args.local_steps < 1:
        raise ValueError("visits and local-steps must be positive")
    device = torch.device("cuda", torch.cuda.current_device())
    config = research_config()
    if args.skip_dense_heatmaps:
        config = replace(config, decode_heatmaps=False)
    construction_context = torch.device(device)
    with construction_context:
        model = AstroMambaHTrainingAdapter(config=config)
    # The strict 1.8 GiB host-RSS cap leaves insufficient headroom for FP32
    # master weights on this Windows/CUDA build. Keep parameters in BF16 and
    # use the caller-selected larger curriculum learning rate so updates are
    # representable; the forward path remains BF16 autocast-compatible.
    model = model.to(device, dtype=torch.bfloat16)
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)

    if args.phase == "synthetic":
        pair = next(
            iter_paired_synthetic_training_batches(
                SyntheticConfig(
                    seed=args.seed,
                    visits=args.visits,
                    local_steps=args.local_steps,
                    raster_height=720,
                    raster_width=1280,
                    wavelength_nm=(450.0, 650.0, 1000.0),
                ),
                sample_count=1,
                device="cpu",
            )
        )
        batch = _split_batch(pair)[args.view]
        # The selected view is a tensor slice, so retain only that view before
        # staging it on CUDA. Keeping the full paired batch here duplicates the
        # multi-step raster in host memory and can breach the RSS cap.
        del pair
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required for the real phase")
        parent = load_parent(args.manifest, exposure_index=args.exposure_index)
        parent_stream = iter_parented_synthetic_training_batches(
            (parent,),
            sample_count=1,
            start_index=args.view,
            device="cpu",
            sequence_summary=args.sequence_summary,
        )
        batch = next(parent_stream)
        # A complete parent sequence is the scientifically correct input, but
        # keeping the FITS-derived NumPy arrays, float32 batch, and CUDA staging
        # buffers alive together can exceed the strict host-RSS cap. The model
        # already trains in BF16, so compact the streamed floating-point input
        # before moving it to CUDA and release the parent generator immediately.
        batch = _cast_batch_floating_tensors(batch, torch.bfloat16)
        del parent_stream, parent
        gc.collect()
    batch = prepare_worker_batch(batch, device)
    gc.collect()
    trainer = BoundedTrainer(
        model,
        config=TrainingConfig(
            device=device,
            max_batches_per_epoch=1,
            amp="auto",
            grad_clip_norm=1.0,
        ),
        optimizer=torch.optim.SGD(model.parameters(), lr=args.learning_rate),
    )
    loss_fn = (
        source_event_loss_fn
        if args.source_event_curriculum
        else event_only_loss_fn
        if args.event_only
        else default_loss_fn
    )
    report = trainer.train_epoch([batch], loss_fn=loss_fn)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    torch.save(
        {"model": model.state_dict(), "parameter_count": report.parameter_count},
        temporary,
    )
    os.replace(temporary, args.checkpoint)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "view": args.view,
                "loss": report.last_loss,
                "finite": report.loss_is_finite,
                "process_rss_bytes": report.process_rss_bytes,
                "rss_within_cap": report.rss_within_cap,
                "peak_gpu_memory_bytes": report.peak_gpu_memory_bytes,
            },
            sort_keys=True,
        )
    )
    return 0 if report.rss_within_cap is not False and report.loss_is_finite else 2


def load_parent(manifest_path: Path, *, exposure_index: int | None = None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = "array_file" in manifest["records"][0]
    if exposure_index is None:
        selected_items = list(manifest["records"])
    else:
        if exposure_index < 0 or exposure_index >= len(manifest["records"]):
            raise IndexError("exposure_index is outside the manifest")
        selected_items = [manifest["records"][exposure_index]]
    source_records = [item["record"] if prepared else item for item in selected_items]
    records = tuple(
        ManifestRecord(
            **{
                **record,
                "wavelength": WavelengthMetadata(**record.get("wavelength", {})),
            }
        )
        for record in source_records
    )
    if prepared:
        paths = {
            record["product_id"]: manifest_path.parent / item["array_file"]
            for item, record in zip(selected_items, source_records)
        }

        def load_arrays(record):
            with np.load(paths[record.product_id], allow_pickle=False) as arrays:
                return {
                    "science": arrays["science"],
                    "uncertainty": arrays["uncertainty"],
                    "dq": arrays["dq"],
                }

        loader = ManifestParentLoader(
            load_arrays,
            time_converter_label="MJD_to_JD_scale_only_local_probe",
            time_converter=lambda record: (
                float(record.observation_start) + 2400000.5,
                float(record.observation_end) + 2400000.5,
            ),
        )
    else:
        paths = {record["product_id"]: manifest_path.parent / record["product_id"] for record in source_records}
        loader = FitsManifestParentLoader(
            paths,
            target_shape=(720, 1280),
            time_converter_label="MJD_to_TDB_scale_only_local_probe",
            time_converter=lambda record: (
                __import__("astropy.time", fromlist=["Time"]).Time(record.observation_start, format="mjd", scale="utc").tdb.jd,
                __import__("astropy.time", fromlist=["Time"]).Time(record.observation_end, format="mjd", scale="utc").tdb.jd,
            ),
        )
    return loader.load(
        records,
        target_id=str(manifest["target"]),
        source_x=float(manifest.get("source_x", 640.0)),
        source_y=float(manifest.get("source_y", 360.0)),
        observation_id=manifest_path.parent.name,
    )


def _cast_batch_floating_tensors(batch, dtype: torch.dtype):
    values = {}
    for name in (
        "raster", "wavelength_tokens", "object_tokens", "geometry",
        "exposure_duration", "coverage_vector", "local_time", "long_time",
        "coverage_map", "source_xy",
    ):
        value = getattr(batch.inputs, name)
        values[name] = (
            value.to(dtype=dtype)
            if isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.dtype not in (torch.float16, torch.bfloat16)
            else value
        )
    return type(batch)(
        inputs=type(batch.inputs)(
            **values,
            wavelength_mask=batch.inputs.wavelength_mask,
            object_mask=batch.inputs.object_mask,
            visit_mask=batch.inputs.visit_mask,
            step_mask=batch.inputs.step_mask,
        ),
        target=batch.target,
        auxiliary_targets=batch.auxiliary_targets,
    )




if __name__ == "__main__":
    raise SystemExit(main())
