import cv2
import json
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from scipy import signal, interpolate

mp_pose = mp.solutions.pose

GAIT_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
    "left_wrist": 15,    "right_wrist": 16,
}

CADENCE_BAND = (0.5, 3.0)   # Hz → 30–180 steps/min

# PD flag thresholds
ARM_ASYMMETRY_THRESH = 1.5
ARM_SWING_LOW_THRESH = 0.06  # normalized to shoulder width
STEP_CV_THRESH       = 4.0   # percent
CADENCE_HIGH_THRESH  = 130   # steps/min (festination)


@dataclass
class SegmentResult:
    label: str             # 'toward', 'away', or 'full'
    start_s: float
    end_s: float
    cadence_spm: float
    step_cv: float         # coefficient of variation of step intervals (%)
    left_arm_swing: float  # normalized to shoulder width
    right_arm_swing: float
    arm_asymmetry: float   # max/min ratio
    step_times: np.ndarray


@dataclass
class GaitAnalysis:
    fps: float
    duration: float
    detection_rate: float
    turnaround_s: float
    segments: list
    time: np.ndarray
    hip_y: np.ndarray          # detrended per segment, for plotting
    left_wrist_x: np.ndarray   # detrended
    right_wrist_x: np.ndarray  # detrended
    skeleton_scale: np.ndarray
    deviation_flags: list


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_gait(video_path: str) -> GaitAnalysis:
    landmarks, vis, fps, frame_count, detection_rate = _track(video_path)

    if detection_rate < 0.3:
        raise ValueError(
            f"Pose detected in only {detection_rate*100:.0f}% of frames — "
            "ensure the full body is visible during walking."
        )

    t = np.arange(frame_count) / fps

    def coord(name, axis, vis_thresh=0.2):
        pos = landmarks[name][:, axis].copy()
        pos[vis[name] < vis_thresh] = np.nan
        valid = ~np.isnan(pos)
        if valid.sum() < fps * 2:
            return np.full(len(pos), np.nan)
        return _fill_nans(t, pos, valid)

    hip_y        = (coord("left_hip", 1) + coord("right_hip", 1)) / 2
    l_shoulder_x = coord("left_shoulder", 0)
    r_shoulder_x = coord("right_shoulder", 0)
    l_shoulder_y = coord("left_shoulder", 1)
    r_shoulder_y = coord("right_shoulder", 1)
    l_ankle_y    = coord("left_ankle", 1, vis_thresh=0.3)
    r_ankle_y    = coord("right_ankle", 1, vis_thresh=0.3)
    l_wrist_x    = coord("left_wrist", 0, vis_thresh=0.5)
    r_wrist_x    = coord("right_wrist", 0, vis_thresh=0.5)

    shoulder_width = np.abs(l_shoulder_x - r_shoulder_x)
    mean_sw = float(np.nanmedian(shoulder_width))

    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
    ankle_y = np.nanmean(np.column_stack([l_ankle_y, r_ankle_y]), axis=1)
    skeleton_scale = np.abs(ankle_y - shoulder_y)

    turnaround_s = _detect_turnaround(skeleton_scale, t, fps)

    if turnaround_s is not None:
        seg_bounds = [(t[0], turnaround_s), (turnaround_s, t[-1])]
        seg_labels = _label_segments(skeleton_scale, t, turnaround_s)
    else:
        seg_bounds = [(t[0], t[-1])]
        seg_labels = ["full"]

    hip_y_det    = _detrend_segments(hip_y,    t, seg_bounds)
    l_wrist_det  = _detrend_segments(l_wrist_x, t, seg_bounds)
    r_wrist_det  = _detrend_segments(r_wrist_x, t, seg_bounds)

    segments = []
    for (start, end), label in zip(seg_bounds, seg_labels):
        mask = (t >= start) & (t <= end)
        if mask.sum() < fps * 2.5:
            continue
        seg = _analyze_segment(
            label, start, end, t[mask],
            hip_y_det[mask], l_wrist_det[mask], r_wrist_det[mask],
            mean_sw, fps,
        )
        if seg is not None:
            segments.append(seg)

    if not segments:
        raise ValueError(
            "Could not extract gait metrics — check that the full body "
            "is visible and the person is walking during the recording."
        )

    return GaitAnalysis(
        fps=fps, duration=frame_count / fps,
        detection_rate=detection_rate,
        turnaround_s=turnaround_s,
        segments=segments,
        time=t,
        hip_y=hip_y_det,
        left_wrist_x=l_wrist_det,
        right_wrist_x=r_wrist_det,
        skeleton_scale=skeleton_scale,
        deviation_flags=_generate_flags(segments),
    )


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def _track(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps < 15:
        print(f"  Warning: low frame rate ({fps:.1f} fps)")
    if total / fps < 4:
        print(f"  Warning: short video ({total/fps:.1f}s) — recommend ≥5s for gait")

    print(f"  Mode: gait  |  {fps:.1f} fps  |  ~{total/fps:.0f}s  ({total} frames)")

    data = {k: [] for k in GAIT_LANDMARKS}
    vis  = {k: [] for k in GAIT_LANDMARKS}
    n = detected = 0

    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            h, w = frame.shape[:2]
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.pose_landmarks:
                detected += 1
                lms = result.pose_landmarks.landmark
                for name, idx in GAIT_LANDMARKS.items():
                    data[name].append([lms[idx].x * w, lms[idx].y * h])
                    vis[name].append(lms[idx].visibility)
            else:
                for name in GAIT_LANDMARKS:
                    data[name].append([np.nan, np.nan])
                    vis[name].append(0.0)

            if n % max(1, int(fps)) == 0:
                print(f"\r  Tracking... {n/fps:.0f}s", end="", flush=True)

    cap.release()
    rate = detected / n if n > 0 else 0.0
    print(f"\r  Tracked {n/fps:.0f}s — pose detected in {rate*100:.0f}% of frames")

    return (
        {k: np.array(v) for k, v in data.items()},
        {k: np.array(v) for k, v in vis.items()},
        fps, n, rate,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _analyze_segment(label, start, end, t, hip_y, l_wrist, r_wrist, shoulder_width_px, fps):
    if np.isnan(hip_y).mean() > 0.5:
        return None

    nyq = fps / 2.0
    lo  = CADENCE_BAND[0] / nyq
    hi  = min(CADENCE_BAND[1], nyq * 0.95) / nyq
    if lo >= hi or len(hip_y) < int(fps * 2):
        return None

    b, a = signal.butter(4, [lo, hi], btype="band")

    try:
        hip_f = signal.filtfilt(b, a, np.nan_to_num(hip_y))
    except ValueError:
        return None

    # Cadence via FFT
    freqs = np.fft.rfftfreq(len(hip_f), 1.0 / fps)
    power = np.abs(np.fft.rfft(hip_f)) ** 2
    fmask = (freqs >= CADENCE_BAND[0]) & (freqs <= CADENCE_BAND[1])
    cadence_spm = float(freqs[fmask][np.argmax(power[fmask])]) * 60 if fmask.any() else np.nan

    # Step regularity via peak detection
    min_dist = max(1, int(fps / CADENCE_BAND[1]))
    peaks, _ = signal.find_peaks(hip_f, distance=min_dist)
    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fps
        step_cv   = float(intervals.std() / intervals.mean() * 100)
        step_times = t[peaks]
    else:
        step_cv    = np.nan
        step_times = np.array([])

    # Arm swing amplitude per side
    def swing(wrist_x):
        valid = ~np.isnan(wrist_x)
        if valid.sum() < fps * 1.5:
            return np.nan
        w = wrist_x.copy()
        w[~valid] = np.nanmean(w)
        try:
            wf = signal.filtfilt(b, a, w)
        except ValueError:
            return np.nan
        return float(np.sqrt(np.mean(wf ** 2)) * 2 * np.sqrt(2) / shoulder_width_px)

    ls = swing(l_wrist)
    rs = swing(r_wrist)

    if np.isnan(ls) or np.isnan(rs) or min(ls, rs) < 1e-6:
        asymmetry = np.nan
    else:
        asymmetry = float(max(ls, rs) / min(ls, rs))

    return SegmentResult(
        label=label, start_s=start, end_s=end,
        cadence_spm=cadence_spm, step_cv=step_cv,
        left_arm_swing=ls, right_arm_swing=rs,
        arm_asymmetry=asymmetry, step_times=step_times,
    )


def _detect_turnaround(scale, t, fps):
    valid = ~np.isnan(scale)
    if valid.sum() < fps * 3:
        return None

    s = scale.copy()
    s[~valid] = np.nanmean(s)

    win    = max(1, int(fps * 1.5))
    smooth = np.convolve(s, np.ones(win) / win, mode="same")

    third   = len(t) // 3
    d_first = np.gradient(smooth[:third]).mean()
    d_last  = np.gradient(smooth[2 * third:]).mean()

    if np.sign(d_first) == np.sign(d_last):
        return None  # no direction reversal

    d   = np.gradient(smooth)
    s_i = len(t) // 5
    e_i = 4 * len(t) // 5
    d_mid = d[s_i:e_i]
    t_mid = t[s_i:e_i]

    changes = np.where(np.diff(np.sign(d_mid)) != 0)[0]
    if len(changes) == 0:
        return None

    center = len(d_mid) // 2
    best   = changes[np.argmin(np.abs(changes - center))]
    return float(t_mid[best])


def _label_segments(scale, t, turnaround_s):
    mid_idx = np.searchsorted(t, turnaround_s)
    window  = max(1, mid_idx // 5)
    pre     = scale[max(0, mid_idx - window):mid_idx]
    pre_trend = np.nanmean(np.gradient(pre[~np.isnan(pre)])) if (~np.isnan(pre)).sum() > 1 else 0
    return ["toward", "away"] if pre_trend > 0 else ["away", "toward"]


def _detrend_segments(arr, t, seg_bounds):
    result = arr.copy()
    for start, end in seg_bounds:
        mask = (t >= start) & (t <= end)
        seg  = result[mask]
        seg_t = t[mask]
        valid = ~np.isnan(seg)
        if valid.sum() < 4:
            continue
        coeffs = np.polyfit(seg_t[valid], seg[valid], 1)
        result[mask] = seg - np.polyval(coeffs, seg_t)
    return result


def _fill_nans(t, v, valid):
    f = interpolate.interp1d(
        t[valid], v[valid], kind="linear",
        bounds_error=False, fill_value=(v[valid][0], v[valid][-1]),
    )
    return f(t)


def _generate_flags(segments):
    flags = []
    for seg in segments:
        lbl = seg.label.upper()

        if not np.isnan(seg.arm_asymmetry) and seg.arm_asymmetry > ARM_ASYMMETRY_THRESH:
            side = "right" if seg.right_arm_swing < seg.left_arm_swing else "left"
            flags.append(
                f"ARM SWING ASYMMETRY ({lbl}): {seg.arm_asymmetry:.2f}× "
                f"({side} arm reduced) — threshold {ARM_ASYMMETRY_THRESH}×"
            )

        for side, val in [("left", seg.left_arm_swing), ("right", seg.right_arm_swing)]:
            if not np.isnan(val) and val < ARM_SWING_LOW_THRESH:
                flags.append(
                    f"REDUCED ARM SWING ({lbl}): {side} {val:.3f} normalized "
                    f"— threshold {ARM_SWING_LOW_THRESH}"
                )

        if not np.isnan(seg.step_cv) and seg.step_cv > STEP_CV_THRESH:
            flags.append(
                f"IRREGULAR GAIT ({lbl}): step interval CV {seg.step_cv:.1f}% "
                f"— threshold {STEP_CV_THRESH}%"
            )

        if not np.isnan(seg.cadence_spm) and seg.cadence_spm > CADENCE_HIGH_THRESH:
            flags.append(
                f"HIGH CADENCE ({lbl}): {seg.cadence_spm:.0f} steps/min "
                f"— possible festination (threshold {CADENCE_HIGH_THRESH})"
            )

    return flags


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_gait_report(analysis: GaitAnalysis, video_path: str):
    print()
    print("=" * 64)
    print("  GAIT ANALYSIS REPORT")
    print("=" * 64)
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Duration : {analysis.duration:.1f} s  |  {analysis.fps:.1f} fps  |  detection {analysis.detection_rate*100:.0f}%")

    if analysis.turnaround_s:
        print(f"  Direction: turnaround at {analysis.turnaround_s:.1f} s")
    else:
        print(f"  Direction: single segment (no turnaround detected)")
    print()

    for seg in analysis.segments:
        print(f"  SEGMENT: {seg.label.upper()}  ({seg.start_s:.1f}–{seg.end_s:.1f} s)")
        if not np.isnan(seg.cadence_spm):
            cv_str = f"  |  step CV {seg.step_cv:.1f}%" if not np.isnan(seg.step_cv) else ""
            print(f"    Cadence         : {seg.cadence_spm:.0f} steps/min{cv_str}")
        if not np.isnan(seg.left_arm_swing) and not np.isnan(seg.right_arm_swing):
            asym = f"  |  asymmetry {seg.arm_asymmetry:.2f}×" if not np.isnan(seg.arm_asymmetry) else ""
            print(f"    Arm swing L / R : {seg.left_arm_swing:.3f}  /  {seg.right_arm_swing:.3f}{asym}")
        print()

    if analysis.deviation_flags:
        print("  FLAGS / DEVIATIONS:")
        for flag in analysis.deviation_flags:
            print(f"    !! {flag}")
    else:
        print("  No deviations detected.")

    print("=" * 64)
    print()


def save_gait_outputs(analysis: GaitAnalysis, output_dir, video_path: str):
    stem = Path(video_path).stem
    out  = Path(output_dir) if output_dir else Path(video_path).parent / f"{stem}_gait"
    out.mkdir(parents=True, exist_ok=True)

    plot_path = _save_plot(analysis, out, stem, video_path)
    json_path = _save_json(analysis, out, stem, video_path)
    print(f"  Plot : {plot_path}")
    print(f"  JSON : {json_path}")


def _save_plot(analysis, out_dir, stem, video_path):
    t = analysis.time

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Gait Analysis — {Path(video_path).name}\n"
        f"Detection: {analysis.detection_rate*100:.0f}%  |  "
        f"Duration: {analysis.duration:.1f}s" +
        (f"  |  Turnaround: {analysis.turnaround_s:.1f}s" if analysis.turnaround_s else ""),
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── Row 0: Hip oscillation with step markers ─────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, analysis.hip_y, lw=0.8, color="steelblue", alpha=0.9)
    ax0.axhline(0, color="gray", lw=0.5, ls="--")
    for seg in analysis.segments:
        for st in seg.step_times:
            ax0.axvline(st, color="limegreen", lw=0.6, alpha=0.5)
    if analysis.turnaround_s:
        ax0.axvline(analysis.turnaround_s, color="crimson", lw=1.5, ls="--",
                    label=f"Turnaround {analysis.turnaround_s:.1f}s")
        ax0.legend(fontsize=8)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Hip vertical\n(detrended, px)")
    ax0.set_title("Hip Oscillation — Step Rhythm  (green = detected steps)")
    ax0.set_xlim(t[0], t[-1])

    # ── Row 1 left: Cadence spectrum ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    hip_clean = np.nan_to_num(analysis.hip_y)
    freqs = np.fft.rfftfreq(len(hip_clean), 1.0 / analysis.fps)
    power = np.abs(np.fft.rfft(hip_clean)) ** 2
    fmask = (freqs >= 0.3) & (freqs <= 3.5)
    ax1.plot(freqs[fmask] * 60, power[fmask], color="steelblue", lw=1.2)
    ax1.axvspan(80, 130, alpha=0.12, color="limegreen", label="Typical range")
    colors_seg = ["darkorange", "crimson", "purple"]
    for i, seg in enumerate(analysis.segments):
        if not np.isnan(seg.cadence_spm):
            ax1.axvline(seg.cadence_spm, lw=1.5, ls="--", color=colors_seg[i % 3],
                        label=f"{seg.label}: {seg.cadence_spm:.0f} spm")
    ax1.set_xlabel("Cadence (steps/min)")
    ax1.set_ylabel("Power")
    ax1.set_title("Cadence Spectrum")
    ax1.legend(fontsize=8)
    ax1.set_xlim(30, 210)

    # ── Row 1 right: Arm swing time series ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t, analysis.left_wrist_x,  lw=0.8, color="steelblue",  alpha=0.85, label="Left wrist")
    ax2.plot(t, analysis.right_wrist_x, lw=0.8, color="darkorange", alpha=0.85, label="Right wrist")
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    if analysis.turnaround_s:
        ax2.axvline(analysis.turnaround_s, color="crimson", lw=1.5, ls="--")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Wrist lateral\n(detrended, px)")
    ax2.set_title("Arm Swing — Wrist Lateral Position")
    ax2.legend(fontsize=8)
    ax2.set_xlim(t[0], t[-1])

    # ── Row 2 left: Arm swing amplitude per segment ───────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(analysis.segments))
    w = 0.35
    left_vals  = [s.left_arm_swing  if not np.isnan(s.left_arm_swing)  else 0 for s in analysis.segments]
    right_vals = [s.right_arm_swing if not np.isnan(s.right_arm_swing) else 0 for s in analysis.segments]
    xlabels    = [s.label for s in analysis.segments]
    ax3.bar(x - w/2, left_vals,  w, label="Left",  color="steelblue",  edgecolor="black", lw=0.5)
    ax3.bar(x + w/2, right_vals, w, label="Right", color="darkorange", edgecolor="black", lw=0.5)
    ax3.axhline(ARM_SWING_LOW_THRESH, color="crimson", ls="--", lw=1,
                label=f"Low threshold ({ARM_SWING_LOW_THRESH})")
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlabels)
    ax3.set_ylabel("Arm swing\n(normalized to shoulder width)")
    ax3.set_title("Arm Swing Amplitude by Segment")
    ax3.legend(fontsize=8)

    # ── Row 2 right: Step interval variability ────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    all_starts, all_ivs = [], []
    for seg in analysis.segments:
        if len(seg.step_times) >= 2:
            ivs = np.diff(seg.step_times)
            all_starts.extend(seg.step_times[:-1].tolist())
            all_ivs.extend(ivs.tolist())

    if all_ivs:
        starts = np.array(all_starts)
        ivs    = np.array(all_ivs)
        lo_iv, hi_iv = 60 / CADENCE_HIGH_THRESH, 60 / 60  # 0.46–1.0 s
        colors = ["limegreen" if lo_iv <= iv <= hi_iv else "crimson" for iv in ivs]
        ax4.bar(starts, ivs, width=ivs * 0.7, color=colors, alpha=0.75, align="edge")
        ax4.axhspan(lo_iv, hi_iv, alpha=0.08, color="limegreen", label="Normal range")
        if analysis.turnaround_s:
            ax4.axvline(analysis.turnaround_s, color="crimson", lw=1.5, ls="--")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Step interval (s)")
        ax4.set_title("Step Interval Over Time\n(green = normal  |  red = irregular)")
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "Insufficient steps\ndetected for interval plot",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=11, color="gray")
        ax4.axis("off")
        ax4.set_title("Step Intervals")

    plot_path = out_dir / f"{stem}_gait.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_json(analysis, out_dir, stem, video_path):
    def seg_dict(s):
        def f(v): return round(float(v), 4) if not np.isnan(v) else None
        return {
            "label":           s.label,
            "start_s":         round(s.start_s, 2),
            "end_s":           round(s.end_s, 2),
            "cadence_spm":     f(s.cadence_spm),
            "step_cv_pct":     f(s.step_cv),
            "left_arm_swing":  f(s.left_arm_swing),
            "right_arm_swing": f(s.right_arm_swing),
            "arm_asymmetry":   f(s.arm_asymmetry),
        }

    summary = {
        "video":              Path(video_path).name,
        "mode":               "gait",
        "duration_s":         round(analysis.duration, 2),
        "fps":                round(analysis.fps, 2),
        "detection_rate_pct": round(analysis.detection_rate * 100, 1),
        "turnaround_s":       round(analysis.turnaround_s, 2) if analysis.turnaround_s else None,
        "segments":           [seg_dict(s) for s in analysis.segments],
        "flags":              analysis.deviation_flags,
        "timestamp":          datetime.now().isoformat(),
    }

    json_path = out_dir / f"{stem}_gait.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return json_path
