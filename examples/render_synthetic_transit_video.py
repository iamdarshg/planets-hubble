"""Render an exact synthetic input pair into a human-viewable video.

The raster panels use the same generator channels consumed by training.  The
bottom panels expose the injected-minus-null signal and source light curves so
the transit is visually inspectable rather than hidden in a tensor file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic import SyntheticConfig, SyntheticGenerator  # noqa: E402


def _visible_rgb(raster: np.ndarray, physical_limits: tuple[float, float]) -> np.ndarray:
    """Compress the six model raster channels into a visible RGB diagnostic.

    This is a display transform, not a replacement for the model input.  R
    carries robust log physical flux, G carries signed noise-scaled residual,
    and B carries inverse uncertainty.  Validity and coverage gate the result.
    The transform is deliberately shared by synthetic and real streams.
    """

    if raster.ndim != 4:
        raise ValueError(f"raster must be [T,C,H,W] or [T,H,W,C], got {raster.shape}")
    if raster.shape[1] == 6:
        raster = np.moveaxis(raster, 1, -1)
    elif raster.shape[-1] != 6:
        raise ValueError(f"raster must expose six channels, got {raster.shape}")
    physical = np.log(np.maximum(raster[..., 0], 1e-6))
    low, high = np.log(max(physical_limits[0], 1e-6)), np.log(max(physical_limits[1], 1e-6))
    red = np.clip((physical - low) / max(high - low, 1e-6), 0.0, 1.0)
    green = 0.5 + 0.5 * np.tanh(np.clip(raster[..., 1], -12.0, 12.0) / 6.0)
    uncertainty_scale = np.maximum(np.nanmedian(raster[..., 2]), 1e-6) * 4.0
    blue = 1.0 - np.clip(raster[..., 2] / uncertainty_scale, 0.0, 1.0)
    gate = np.clip(raster[..., 3] * raster[..., 5], 0.0, 1.0)
    return np.clip(np.stack((red, green, blue), axis=-1) * gate[..., None], 0.0, 1.0)


def _residual_rgb(delta_residual: np.ndarray) -> np.ndarray:
    signed = 0.5 + 0.5 * np.tanh(np.clip(delta_residual, -12.0, 12.0) / 4.0)
    return np.stack((1.0 - signed, np.full_like(signed, 0.18), signed), axis=-1)


def _config(seed: int) -> SyntheticConfig:
    rng = np.random.default_rng(seed * 1_000_003)
    source_x, source_y = (float(value) for value in rng.uniform(0.14, 0.86, size=2))
    if 0.42 <= source_x <= 0.58 and 0.42 <= source_y <= 0.58:
        source_x = 0.22 if source_x < 0.5 else 0.78
    return SyntheticConfig(
        seed=seed,
        visits=1,
        local_steps=48,
        raster_height=32,
        raster_width=32,
        local_step_spacing_days=0.005,
        wavelength_nm=(450.0, 550.0, 650.0, 800.0, 1000.0),
        field_star_count=10,
        field_planet_probability=0.45,
        field_star_flux_ratio_min=0.03,
        field_star_flux_ratio_max=0.30,
        field_star_min_separation_pixels=3.0,
        barycentric_tdb_offset_seconds=120.0,
        light_time_correction_seconds=480.0,
        apparent_position_shift_arcsec=0.03,
        stellar_radial_velocity_mps=20_000.0,
        barycentric_radial_velocity_mps=29_000.0,
        gravitational_redshift_mps=636.0,
        orbital_radial_velocity_amplitude_mps=150.0,
        source_x=source_x,
        source_y=source_y,
        hst_thermal_breathing_amplitude=0.002,
        hst_focus_psf_amplitude=0.08,
        pointing_jitter_pixels=0.12,
        drift_pixels_per_visit=0.04,
        roll_amplitude_deg=0.15,
        geometric_distortion_amplitude=0.001,
        pam_gradient_amplitude=0.003,
        radiation_hot_pixel_rate=0.0005,
        cosmic_ray_rate=0.0002,
        shutter_artifact_amplitude=0.001,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/synthetic-visuals"))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _config(args.seed)
    bundle = SyntheticGenerator(config).generate()
    null_model = bundle.null.raster[0]
    injected_model = bundle.injected.raster[0]
    all_physical = np.concatenate((null_model[:, 0].ravel(), injected_model[:, 0].ravel()))
    physical_limits = (
        float(np.percentile(all_physical, 1.0)),
        float(np.percentile(all_physical, 99.5)),
    )
    null_rgb = _visible_rgb(null_model, physical_limits)
    injected_rgb = _visible_rgb(injected_model, physical_limits)
    residual_rgb = _residual_rgb(injected_model[:, 1] - null_model[:, 1])
    residual = injected_model[:, 0] - null_model[:, 0]
    times = bundle.timestamps_mid_bjd_tdb[0]
    observed_wavelengths = np.asarray(config.wavelength_nm) * bundle.relativity_terms["doppler_factor"][0, 0]
    injected_wavelength_curves = bundle.injected.wavelength_tokens[0, :, :, 1]
    null_wavelength_curves = bundle.null.wavelength_tokens[0, :, :, 1]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    figure.suptitle("Synthetic transit input: six model raster channels compressed to RGB; 10-star field")
    image_null = axes[0, 0].imshow(
        null_rgb[0], interpolation="nearest", animated=True
    )
    image_injected = axes[0, 1].imshow(
        injected_rgb[0], interpolation="nearest", animated=True
    )
    image_residual = axes[0, 2].imshow(
        residual_rgb[0],
        interpolation="nearest",
        animated=True,
    )
    for axis, title in zip(
        axes[0], ("Null model-channel RGB", "Injected model-channel RGB", "Injected - null residual"), strict=True
    ):
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    for star_index, star in enumerate(bundle.source_metadata["field_stars"]):
        star_x = star["x"] * (config.raster_width - 1)
        star_y = star["y"] * (config.raster_height - 1)
        for axis, color in zip(axes[0], ("cyan", "cyan", "white"), strict=True):
            axis.plot(
                star_x,
                star_y,
                marker="+" if star_index == 0 else ".",
                color=color,
                markersize=9 if star_index == 0 else 4,
                alpha=1.0 if star_index == 0 else 0.7,
            )

    curve_axis = axes[1, 0]
    for wavelength_index, wavelength in enumerate(observed_wavelengths):
        curve_axis.plot(
            times - times[0],
            injected_wavelength_curves[:, wavelength_index],
            label=f"{wavelength:.1f} nm",
        )
    curve_axis.set_title("Injected source light curves")
    curve_axis.set_xlabel("Time since first frame (days)")
    curve_axis.set_ylabel("Physical ratio")
    curve_axis.legend(fontsize=7, ncol=2)
    curve_axis.set_ylim(0.965, 1.02)

    diff_axis = axes[1, 1]
    diff_lines = [
        diff_axis.plot([], [], label=f"{wavelength:.1f} nm")[0]
        for wavelength in observed_wavelengths
    ]
    diff_axis.set_title("Source transit decrement")
    diff_axis.set_xlabel("Time since first frame (days)")
    diff_axis.set_ylabel("Injected - null")
    diff_axis.set_xlim(float((times - times[0]).min()), float((times - times[0]).max()))
    diff_axis.set_ylim(float(np.min(residual)) * 1.1, 0.001)
    diff_axis.legend(fontsize=7, ncol=2)

    info_axis = axes[1, 2]
    info_axis.axis("off")
    info_axis.set_title("Simulation metadata")
    info = (
        f"seed: {config.seed}\n"
        f"event: {bundle.injected.labels.event_type}\n"
        f"target source xy: ({config.source_x:.3f}, {config.source_y:.3f})\n"
        f"field stars: {config.field_star_count}; planet prior: {config.field_planet_probability:.2f}\n"
        f"depth: {float(bundle.injected.labels.transit_depth.max()):.5f}\n"
        f"BJD_TDB offset: {config.barycentric_tdb_offset_seconds:.1f} s\n"
        f"light-time correction: {config.light_time_correction_seconds:.1f} s\n"
        f"radial velocity: {float(bundle.relativity_terms['radial_velocity_mps'][0, 0]):.1f} m/s\n"
        f"Doppler factor: {float(bundle.relativity_terms['doppler_factor'][0, 0]):.9f}\n"
        f"observed wavelengths:\n{np.array2string(observed_wavelengths, precision=5)}\n"
        "RGB: log physical | signed residual | inverse uncertainty\n"
        f"relativity scope: {bundle.relativity_metadata['scope']}"
    )
    info_axis.text(0.02, 0.98, info, va="top", family="monospace", fontsize=8, wrap=True)

    time_text = figure.text(0.5, 0.01, "", ha="center", family="monospace")

    def update(frame: int):
        image_null.set_data(null_rgb[frame])
        image_injected.set_data(injected_rgb[frame])
        image_residual.set_data(residual_rgb[frame])
        for wavelength_index, line in enumerate(diff_lines):
            line.set_data(
                times[: frame + 1] - times[0],
                injected_wavelength_curves[: frame + 1, wavelength_index]
                - null_wavelength_curves[: frame + 1, wavelength_index],
            )
        time_text.set_text(
            f"frame {frame + 1:02d}/{len(times)}   BJD_TDB={times[frame]:.8f}   "
            f"observed RV={float(bundle.relativity_terms['radial_velocity_mps'][0, frame]):.2f} m/s"
        )
        return image_null, image_injected, image_residual, *diff_lines, time_text

    animation = FuncAnimation(figure, update, frames=len(times), interval=1000 / args.fps, blit=False)
    mp4_path = args.output_dir / "synthetic_transit_input.mp4"
    gif_path = args.output_dir / "synthetic_transit_input.gif"
    render_dpi = 80
    if shutil.which("ffmpeg"):
        animation.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=2200), dpi=render_dpi)
        video_path = mp4_path
    else:
        animation.save(gif_path, writer=PillowWriter(fps=args.fps), dpi=render_dpi)
        video_path = gif_path
    figure.savefig(args.output_dir / "synthetic_transit_contact_sheet.png", dpi=160)
    metadata = {
        "video": str(video_path),
        "contact_sheet": str(args.output_dir / "synthetic_transit_contact_sheet.png"),
        "seed": config.seed,
        "frames": len(times),
        "display_resolution": [1280, 720],
        "rgb_mapping": "R=log physical flux, G=signed noise-scaled residual, B=inverse uncertainty; validity*coverage gate",
        "raster_shape": list(bundle.injected.raster.shape),
        "event_type": bundle.injected.labels.event_type,
        "transit_depth": bundle.injected.labels.transit_depth.tolist(),
        "relativity_metadata": bundle.relativity_metadata,
        "relativity_terms": {
            name: value.tolist() for name, value in bundle.relativity_terms.items()
        },
        "observed_wavelengths_nm_first_frame": observed_wavelengths.tolist(),
        "config": {
            "barycentric_tdb_offset_seconds": config.barycentric_tdb_offset_seconds,
            "light_time_correction_seconds": config.light_time_correction_seconds,
            "apparent_position_shift_arcsec": config.apparent_position_shift_arcsec,
            "stellar_radial_velocity_mps": config.stellar_radial_velocity_mps,
            "barycentric_radial_velocity_mps": config.barycentric_radial_velocity_mps,
            "gravitational_redshift_mps": config.gravitational_redshift_mps,
            "orbital_radial_velocity_amplitude_mps": config.orbital_radial_velocity_amplitude_mps,
            "source_x": config.source_x,
            "source_y": config.source_y,
            "field_star_count": config.field_star_count,
            "field_planet_probability": config.field_planet_probability,
        },
    }
    (args.output_dir / "synthetic_transit_video.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
