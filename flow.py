"""
Optical flow tremor analysis — for difficult camera angles where MediaPipe
cannot detect hand landmarks.

The user draws a box around the area of interest (knuckles, wrist, etc.)
on the first frame. Dense optical flow tracks pixel motion in that region
frame-by-frame. FFT of the motion signal extracts tremor frequency.

Works at any camera angle. Requires no hand detection.
"""

import csv
import cv2
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from scipy import signal

PD_LOW  = 3.0
PD_HIGH = 6.0
ANALYSIS_BAND = (2.0, 15.0)


@dataclass
class FlowAnalysis:
    fps: float
    duration: float
    roi: tuple               # (x, y, w, h) in pixels
    time: np.ndarray
    flow_mag: np.ndarray     # mean motion magnitude in ROI per frame
    flow_y: np.ndarray       # mean vertical motion in ROI per frame
    dominant_freq: float
    amplitude: float         # RMS of filtered magnitude signal
    tremor_index: float      # fraction of power in PD band
    in_pd_range: bool
    freqs: np.ndarray
    power: np.ndarray
    spectrogram_t: np.ndarray
    spectrogram_f: np.ndarray
    spectrogram_p: np.ndarray
    first_frame: np.ndarray  # stored for plot
    deviation_flags: list


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_flow(video_path: str, roi: tuple = None,
                 manifest_path: str = None) -> FlowAnalysis:
    """
    roi: (x, y, w, h) or None for interactive selection.
    manifest_path: if provided and roi was interactively selected,
                   offers to save coordinates back to the manifest.
    """
    if roi is None:
        roi = _select_roi(video_path)
        if manifest_path:
            _offer_save_roi(manifest_path, video_path, roi)

    print(f"  ROI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
    flow_mag, flow_y, first_frame, fps, n_frames = _compute_flow(video_path, roi)

    t = np.arange(len(flow_mag)) / fps

    dominant_freq, amplitude, tremor_index, freqs, power = _analyze_signal(
        flow_mag, fps
    )

    nperseg = max(16, min(int(fps * 2), len(flow_mag) // 2))
    f_spec, t_spec, Sxx = signal.spectrogram(
        flow_mag, fs=fps, nperseg=nperseg, noverlap=int(nperseg * 0.75),
        window="hann",
    )

    flags = _generate_flags(dominant_freq, tremor_index, fps, f_spec, Sxx)

    return FlowAnalysis(
        fps=fps, duration=n_frames / fps, roi=roi,
        time=t, flow_mag=flow_mag, flow_y=flow_y,
        dominant_freq=dominant_freq, amplitude=amplitude,
        tremor_index=tremor_index, in_pd_range=PD_LOW <= dominant_freq <= PD_HIGH,
        freqs=freqs, power=power,
        spectrogram_t=t_spec, spectrogram_f=f_spec, spectrogram_p=Sxx,
        first_frame=first_frame,
        deviation_flags=flags,
    )


# ---------------------------------------------------------------------------
# ROI selection
# ---------------------------------------------------------------------------

def _select_roi(video_path: str) -> tuple:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame for ROI selection.")

    print("  Draw a box around the tremor area, then press ENTER or SPACE.")
    print("  Press C to cancel.")

    # Scale down tall portrait videos for the selection window
    h, w = frame.shape[:2]
    scale = min(1.0, 900 / h, 1200 / w)
    display = cv2.resize(frame, (int(w * scale), int(h * scale)))

    roi_scaled = cv2.selectROI(
        "Select tremor region — ENTER to confirm, C to cancel",
        display, fromCenter=False, showCrosshair=True,
    )
    cv2.destroyAllWindows()

    if roi_scaled == (0, 0, 0, 0):
        raise ValueError("No ROI selected — cancelled.")

    # Scale coordinates back to original frame size
    x = int(roi_scaled[0] / scale)
    y = int(roi_scaled[1] / scale)
    rw = int(roi_scaled[2] / scale)
    rh = int(roi_scaled[3] / scale)
    return (x, y, rw, rh)


def _offer_save_roi(manifest_path: str, video_path: str, roi: tuple):
    resp = input(f"  Save ROI to manifest for future reruns? [y/N] ").strip().lower()
    if resp == "y":
        save_roi_to_manifest(manifest_path, video_path, roi)
        print(f"  ROI saved to {manifest_path}")


def save_roi_to_manifest(manifest_path: str, video_path: str, roi: tuple):
    """Write roi coordinates into the manifest CSV for the matching video row."""
    roi_str = f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}"
    target = str(Path(video_path).resolve())

    rows = []
    fieldnames = None
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        if "roi" not in fieldnames:
            fieldnames.append("roi")
        for row in reader:
            if str(Path(row["path"]).resolve()) == target:
                row["roi"] = roi_str
            rows.append(row)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_roi(roi_str: str) -> tuple:
    """Parse 'x,y,w,h' string into a (x, y, w, h) int tuple."""
    parts = [int(v.strip()) for v in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must be 'x,y,w,h' — got: {roi_str}")
    return tuple(parts)


# ---------------------------------------------------------------------------
# Optical flow computation
# ---------------------------------------------------------------------------

def _compute_flow(video_path: str, roi: tuple):
    x, y, w, h = roi
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps < 15:
        print(f"  Warning: low frame rate ({fps:.1f} fps)")
    print(f"  Mode: flow  |  {fps:.1f} fps  |  ~{total/fps:.0f}s  ({total} frames)")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read video frames.")

    first_frame = frame.copy()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]

    mag_series = []
    y_series   = []
    n = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        mag_series.append(float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()))
        y_series.append(float(flow[..., 1].mean()))

        prev_gray = curr_gray

        if n % max(1, int(fps)) == 0:
            print(f"\r  Computing flow... {n/fps:.0f}s", end="", flush=True)

    cap.release()
    print(f"\r  Flow computed — {n} frames")

    return (
        np.array(mag_series),
        np.array(y_series),
        first_frame, fps, n,
    )


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------

def _analyze_signal(flow_mag: np.ndarray, fps: float):
    nyq = fps / 2.0
    lo  = ANALYSIS_BAND[0] / nyq
    hi  = min(ANALYSIS_BAND[1], nyq * 0.95) / nyq

    b, a = signal.butter(4, [lo, hi], btype="band")
    try:
        filtered = signal.filtfilt(b, a, flow_mag - flow_mag.mean())
    except ValueError:
        filtered = flow_mag - flow_mag.mean()

    freqs = np.fft.rfftfreq(len(filtered), 1.0 / fps)
    power = np.abs(np.fft.rfft(filtered)) ** 2

    band_mask = (freqs >= ANALYSIS_BAND[0]) & (freqs <= ANALYSIS_BAND[1])
    dominant_freq = float(freqs[band_mask][np.argmax(power[band_mask])]) \
                    if band_mask.any() else np.nan

    pd_mask  = (freqs >= PD_LOW) & (freqs <= PD_HIGH)
    tremor_index = float(power[pd_mask].sum() / (power[band_mask].sum() + 1e-12))
    amplitude    = float(np.sqrt(np.mean(filtered ** 2)) * 2 * np.sqrt(2))

    return dominant_freq, amplitude, tremor_index, freqs, power


def _generate_flags(dominant_freq, tremor_index, fps, f_spec, Sxx):
    flags = []

    if np.isnan(dominant_freq):
        flags.append("FREQUENCY NOT DETECTED — signal too weak or video too short.")
        return flags

    if not (PD_LOW <= dominant_freq <= PD_HIGH):
        flags.append(
            f"FREQUENCY OUT OF PD RANGE: {dominant_freq:.2f} Hz "
            f"(expected {PD_LOW}–{PD_HIGH} Hz)"
        )
    if tremor_index < 0.3:
        flags.append(
            f"LOW PD-BAND POWER: {tremor_index*100:.0f}% — "
            "motion may not be tremor-dominated"
        )

    band_mask = (f_spec >= ANALYSIS_BAND[0]) & (f_spec <= ANALYSIS_BAND[1])
    if band_mask.any() and Sxx.shape[1] >= 4:
        dom_per_win = f_spec[band_mask][np.argmax(Sxx[band_mask], axis=0)]
        if dom_per_win.std() > 0.8:
            flags.append(
                f"FREQUENCY INSTABILITY: {dom_per_win.mean():.1f} ± "
                f"{dom_per_win.std():.1f} Hz across recording windows"
            )

    return flags


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_flow_report(analysis: FlowAnalysis, video_path: str):
    print()
    print("=" * 64)
    print("  OPTICAL FLOW TREMOR REPORT")
    print("=" * 64)
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Duration : {analysis.duration:.1f} s  |  {analysis.fps:.1f} fps")
    print(f"  ROI      : x={analysis.roi[0]} y={analysis.roi[1]} "
          f"w={analysis.roi[2]} h={analysis.roi[3]}")
    print()
    range_tag = "IN PD RANGE" if analysis.in_pd_range else "OUTSIDE PD RANGE"
    print(f"  Dominant freq   : {analysis.dominant_freq:.2f} Hz   [{range_tag}]")
    print(f"  Amplitude       : {analysis.amplitude:.4f} px/frame (mean motion)")
    print(f"  PD-band power   : {analysis.tremor_index * 100:.0f}%")
    print()
    if analysis.deviation_flags:
        print("  FLAGS / DEVIATIONS:")
        for flag in analysis.deviation_flags:
            print(f"    !! {flag}")
    else:
        print("  No deviations detected.")
    print("=" * 64)
    print()


def save_flow_outputs(analysis: FlowAnalysis, output_dir, video_path: str):
    stem = Path(video_path).stem
    out  = Path(output_dir) if output_dir else Path(video_path).parent / f"{stem}_flow"
    out.mkdir(parents=True, exist_ok=True)
    plot_path = _save_plot(analysis, out, stem, video_path)
    json_path = _save_json(analysis, out, stem, video_path)
    print(f"  Plot : {plot_path}")
    print(f"  JSON : {json_path}")


def _save_plot(analysis, out_dir, stem, video_path):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Optical Flow Tremor — {Path(video_path).name}\n"
        f"{analysis.dominant_freq:.2f} Hz  |  "
        f"{'IN PD RANGE' if analysis.in_pd_range else 'OUTSIDE PD RANGE'}  |  "
        f"PD-band power {analysis.tremor_index*100:.0f}%",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── Row 0: First frame with ROI ───────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    frame_rgb = cv2.cvtColor(analysis.first_frame, cv2.COLOR_BGR2RGB)
    x, y, w, h = analysis.roi
    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 80, 80), 3)
    ax0.imshow(frame_rgb)
    ax0.axis("off")
    ax0.set_title("ROI Selection (red box)")

    # ── Row 0: Flow magnitude time series ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(analysis.time, analysis.flow_mag, lw=0.8, color="steelblue", alpha=0.9)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Motion magnitude\n(px/frame)")
    ax1.set_title("Motion in ROI Over Time")
    ax1.set_xlim(analysis.time[0], analysis.time[-1])

    # ── Row 1 left: Filtered signal ───────────────────────────────────────────
    nyq = analysis.fps / 2.0
    lo  = ANALYSIS_BAND[0] / nyq
    hi  = min(ANALYSIS_BAND[1], nyq * 0.95) / nyq
    b, a = signal.butter(4, [lo, hi], btype="band")
    try:
        filtered = signal.filtfilt(b, a, analysis.flow_mag - analysis.flow_mag.mean())
    except ValueError:
        filtered = analysis.flow_mag - analysis.flow_mag.mean()

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(analysis.time, filtered, lw=0.8, color="steelblue", alpha=0.9)
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Filtered motion\n(2–15 Hz bandpass)")
    ax2.set_title("Tremor Signal — Bandpass Filtered")
    ax2.set_xlim(analysis.time[0], analysis.time[-1])

    # ── Row 2 left: FFT spectrum ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    fmask = (analysis.freqs >= 0.5) & (analysis.freqs <= 15)
    ax3.semilogy(analysis.freqs[fmask], analysis.power[fmask],
                 color="steelblue", lw=1.2)
    ax3.axvspan(PD_LOW, PD_HIGH, alpha=0.15, color="limegreen",
                label=f"PD range ({PD_LOW}–{PD_HIGH} Hz)")
    ax3.axvline(analysis.dominant_freq, color="crimson", lw=1.5, ls="--",
                label=f"Peak {analysis.dominant_freq:.2f} Hz")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power (log)")
    ax3.set_title("Frequency Spectrum")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0.5, 15)

    # ── Row 2 right: Spectrogram ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    fmask2 = (analysis.spectrogram_f >= 0.5) & (analysis.spectrogram_f <= 15)
    Sdb = 10 * np.log10(analysis.spectrogram_p[fmask2] + 1e-12)
    im  = ax4.pcolormesh(analysis.spectrogram_t,
                         analysis.spectrogram_f[fmask2], Sdb,
                         shading="gouraud", cmap="viridis")
    ax4.axhspan(PD_LOW, PD_HIGH, alpha=0.25, color="limegreen", label="PD range")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_title("Spectrogram")
    ax4.legend(fontsize=8)
    plt.colorbar(im, ax=ax4, label="dB")

    plot_path = out_dir / f"{stem}_flow.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_json(analysis, out_dir, stem, video_path):
    def f(v):
        try: return round(float(v), 4) if not np.isnan(v) else None
        except: return None

    summary = {
        "video":            Path(video_path).name,
        "mode":             "flow",
        "duration_s":       round(analysis.duration, 2),
        "fps":              round(analysis.fps, 2),
        "roi":              list(analysis.roi),
        "dominant_freq_hz": f(analysis.dominant_freq),
        "amplitude":        f(analysis.amplitude),
        "pd_band_power_pct": f(analysis.tremor_index * 100),
        "in_pd_range":      analysis.in_pd_range,
        "flags":            analysis.deviation_flags,
        "timestamp":        datetime.now().isoformat(),
    }

    json_path = out_dir / f"{stem}_flow.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return json_path
