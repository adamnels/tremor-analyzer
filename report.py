import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive — saves to file only
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from scipy.signal import hilbert

PD_LOW = 3.0
PD_HIGH = 6.0

REFERENCE_LABELS = {
    "hands": "wrist-to-middle-MCP (~4 cm)",
    "feet":  "hip width (~30 cm)",
    "face":  "outer eye-to-eye (~6 cm)",
}


def generate_report(analysis, video_path: str):
    p = analysis.primary

    print()
    print("=" * 64)
    print("  TREMOR ANALYSIS REPORT")
    print("=" * 64)
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Mode     : {analysis.mode}  |  {analysis.fps:.1f} fps  |  {analysis.duration:.1f} s  |  detection {analysis.detection_rate*100:.0f}%")
    print()
    print(f"  PRIMARY LANDMARK : {p.name.upper().replace('_', ' ')}")
    range_tag = "IN PD RANGE" if p.in_pd_range else "OUTSIDE PD RANGE"
    print(f"    Dominant freq   : {p.dominant_freq:.2f} Hz   [{range_tag}]")
    ref_label = REFERENCE_LABELS.get(analysis.mode, "reference distance")
    print(f"    Amplitude (p-p) : {p.amplitude:.3f} × {ref_label}")
    print(f"    PD-band power   : {p.tremor_index * 100:.0f}% of total tremor power")
    print()

    if len(analysis.landmarks) > 1:
        print("  ALL LANDMARKS:")
        for lm in analysis.landmarks:
            tag = "PD" if lm.in_pd_range else "!!"
            print(
                f"    [{tag}] {lm.name:<22}  "
                f"{lm.dominant_freq:5.2f} Hz   amp={lm.amplitude:.3f}"
            )
        print()

    if analysis.deviation_flags:
        print("  FLAGS / DEVIATIONS:")
        for flag in analysis.deviation_flags:
            print(f"    !! {flag}")
    else:
        print("  No deviations detected.")

    print("=" * 64)
    print()


def save_outputs(analysis, output_dir, video_path: str):
    stem = Path(video_path).stem
    out = (
        Path(output_dir)
        if output_dir
        else Path(video_path).parent / f"{stem}_tremor"
    )
    out.mkdir(parents=True, exist_ok=True)

    plot_path = _save_plot(analysis, out, stem, video_path)
    json_path = _save_json(analysis, out, stem, video_path)

    print(f"  Plot : {plot_path}")
    print(f"  JSON : {json_path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _save_plot(analysis, out_dir: Path, stem: str, video_path: str) -> Path:
    p = analysis.primary

    fig = plt.figure(figsize=(14, 10))
    range_tag = "IN PD RANGE" if p.in_pd_range else "OUTSIDE PD RANGE"
    fig.suptitle(
        f"Tremor Analysis — {Path(video_path).name}\n"
        f"Mode: {analysis.mode}  |  Primary: {p.name.replace('_', ' ')}  |  "
        f"{p.dominant_freq:.2f} Hz  [{range_tag}]",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── Row 0: Displacement time series ─────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(p.time, p.displacement, lw=0.7, color="steelblue", alpha=0.9)
    ax0.axhline(0, color="gray", lw=0.5, ls="--")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Displacement\n(normalized)")
    ax0.set_title("Primary Landmark Displacement (2–15 Hz bandpass)")
    ax0.set_xlim(p.time[0], p.time[-1])

    # ── Row 1 left: FFT power spectrum ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    freq_mask = (p.freqs >= 0.5) & (p.freqs <= 15)
    ax1.semilogy(p.freqs[freq_mask], p.power[freq_mask], color="steelblue", lw=1.2)
    ax1.axvspan(PD_LOW, PD_HIGH, alpha=0.15, color="limegreen", label=f"PD range ({PD_LOW}–{PD_HIGH} Hz)")
    ax1.axvline(p.dominant_freq, color="crimson", lw=1.5, ls="--",
                label=f"Peak {p.dominant_freq:.2f} Hz")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power (log)")
    ax1.set_title("Frequency Spectrum")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0.5, 15)

    # ── Row 1 right: Spectrogram ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    f_mask = (p.spectrogram_f >= 0.5) & (p.spectrogram_f <= 15)
    Sxx_db = 10 * np.log10(p.spectrogram_p[f_mask] + 1e-12)
    im = ax2.pcolormesh(
        p.spectrogram_t, p.spectrogram_f[f_mask], Sxx_db,
        shading="gouraud", cmap="viridis",
    )
    ax2.axhspan(PD_LOW, PD_HIGH, alpha=0.25, color="limegreen", label="PD range")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Spectrogram")
    ax2.legend(fontsize=8)
    plt.colorbar(im, ax=ax2, label="dB")

    # ── Row 2 left: Dominant frequency over time ─────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    f_s = p.spectrogram_f
    s = p.spectrogram_p
    band_mask = (f_s >= 0.5) & (f_s <= 15)
    if band_mask.any() and s.shape[1] >= 2:
        dom_f = f_s[band_mask][np.argmax(s[band_mask], axis=0)]
        t_s = p.spectrogram_t
        for i in range(len(t_s) - 1):
            color = "green" if PD_LOW <= dom_f[i] <= PD_HIGH else "crimson"
            ax3.plot(t_s[i : i + 2], dom_f[i : i + 2], color=color, lw=2)
        ax3.axhspan(PD_LOW, PD_HIGH, alpha=0.1, color="limegreen")
        ax3.set_ylim(0, 15)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title("Dominant Frequency Over Time\n(green = PD range  |  red = deviation)")

    # ── Row 2 right: Amplitude envelope over time ────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    envelope = np.abs(hilbert(p.displacement))
    win = max(1, int(analysis.fps * 0.5))
    smooth = np.convolve(envelope, np.ones(win) / win, mode="same")
    ax4.plot(p.time, smooth, color="darkorange", lw=1.2)
    ax4.fill_between(p.time, 0, smooth, alpha=0.25, color="darkorange")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude\n(normalized)")
    ax4.set_title("Amplitude Envelope Over Time")
    ax4.set_xlim(p.time[0], p.time[-1])

    # ── Optional: per-landmark comparison ───────────────────────────────────
    if len(analysis.landmarks) > 1:
        fig2, (ax_f, ax_a) = plt.subplots(1, 2, figsize=(10, 4))
        fig2.suptitle("Per-Landmark Comparison", fontsize=11, fontweight="bold")

        names = [lm.name.replace("_", "\n") for lm in analysis.landmarks]
        freqs_lm = [lm.dominant_freq for lm in analysis.landmarks]
        amps_lm = [lm.amplitude for lm in analysis.landmarks]
        colors = ["green" if lm.in_pd_range else "crimson" for lm in analysis.landmarks]

        bars_f = ax_f.bar(names, freqs_lm, color=colors, edgecolor="black", lw=0.6)
        ax_f.axhspan(PD_LOW, PD_HIGH, alpha=0.12, color="limegreen")
        ax_f.set_ylabel("Dominant Frequency (Hz)")
        ax_f.set_title("Frequency (green = in PD range)")
        ax_f.set_ylim(0, 16)
        for bar, val in zip(bars_f, freqs_lm):
            ax_f.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                      f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        bars_a = ax_a.bar(names, amps_lm, color="steelblue", edgecolor="black", lw=0.6)
        ax_a.set_ylabel("Amplitude (normalized)")
        ax_a.set_title("Amplitude")
        for bar, val in zip(bars_a, amps_lm):
            ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig2.tight_layout()
        lm_path = out_dir / f"{stem}_landmarks.png"
        fig2.savefig(lm_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

    plot_path = out_dir / f"{stem}_tremor.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def _save_json(analysis, out_dir: Path, stem: str, video_path: str) -> Path:
    def lm_to_dict(lm):
        return {
            "name": lm.name,
            "dominant_freq_hz": round(lm.dominant_freq, 3),
            "amplitude_normalized": round(lm.amplitude, 4),
            "pd_band_power_pct": round(lm.tremor_index * 100, 1),
            "in_pd_range": lm.in_pd_range,
        }

    summary = {
        "video": Path(video_path).name,
        "mode": analysis.mode,
        "duration_s": round(analysis.duration, 2),
        "fps": round(analysis.fps, 2),
        "detection_rate_pct": round(analysis.detection_rate * 100, 1),
        "amplitude_reference": REFERENCE_LABELS.get(analysis.mode, "reference distance"),
        "primary": lm_to_dict(analysis.primary),
        "all_landmarks": [lm_to_dict(lm) for lm in analysis.landmarks],
        "flags": analysis.deviation_flags,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = out_dir / f"{stem}_tremor.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    return json_path
