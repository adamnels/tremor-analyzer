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
from scipy import signal
from scipy.signal import savgol_filter

mp_hands = mp.solutions.hands

THUMB_TIP  = 4
INDEX_TIP  = 8
HAND_REF   = (0, 9)   # wrist → middle MCP (~4 cm)

# Clinical thresholds
RATE_LOW_THRESH            = 3.5   # Hz — below this is bradykinetic
AMPLITUDE_DECREMENT_THRESH = 25.0  # % — UPDRS-significant decrement
RATE_DECREMENT_THRESH      = 25.0  # %
ARREST_MULTIPLIER          = 2.0   # inter-tap > N × median = arrest
RHYTHM_CV_THRESH           = 20.0  # %


@dataclass
class TapAnalysis:
    hand: str               # 'Left' or 'Right'
    fps: float
    duration: float
    detection_rate: float
    time: np.ndarray
    distance: np.ndarray    # thumb-to-index distance, normalized, smoothed
    tap_times: np.ndarray   # one per tap (fingertips close)
    tap_amplitudes: np.ndarray  # opening width per tap (normalized)
    mean_rate_hz: float
    mean_amplitude: float
    rhythm_cv: float        # CV of inter-tap intervals (%)
    early_rate_hz: float    # first third
    mid_rate_hz: float
    late_rate_hz: float
    early_amplitude: float
    mid_amplitude: float
    late_amplitude: float
    rate_decrement_pct: float       # (early - late) / early × 100
    amplitude_decrement_pct: float
    arrest_count: int
    arrest_times: np.ndarray
    deviation_flags: list


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_taps(video_path: str) -> TapAnalysis:
    distance, ref_size, hand_label, fps, frame_count, detection_rate = _track(video_path)

    if detection_rate < 0.5:
        raise ValueError(
            f"Hand detected in only {detection_rate*100:.0f}% of frames — "
            "ensure the hand fills most of the frame with good lighting."
        )

    t = np.arange(frame_count) / fps

    # Normalize and smooth
    dist_norm = distance / ref_size
    win = max(3, min(11, int(fps * 0.1)) | 1)  # odd window, ~100 ms
    dist_smooth = savgol_filter(
        np.nan_to_num(dist_norm, nan=np.nanmean(dist_norm)),
        window_length=win, polyorder=2,
    )

    tap_idx, amplitudes, intervals = _detect_taps(dist_smooth, fps)

    if len(tap_idx) < 5:
        raise ValueError(
            f"Only {len(tap_idx)} taps detected — ensure the patient taps "
            "clearly and the video is long enough (recommend 10 s)."
        )

    tap_times = t[tap_idx]
    tap_amps  = np.array(amplitudes)

    # Inter-tap intervals
    intervals_s = np.diff(tap_times)
    mean_rate   = 1.0 / intervals_s.mean() if len(intervals_s) > 0 else np.nan
    rhythm_cv   = float(intervals_s.std() / intervals_s.mean() * 100) if len(intervals_s) > 1 else np.nan

    # Arrests
    median_iv    = np.median(intervals_s)
    arrest_mask  = intervals_s > (ARREST_MULTIPLIER * median_iv)
    arrest_times = tap_times[:-1][arrest_mask]

    # Thirds — based on tap events, not absolute time
    n = len(tap_idx)
    t1, t2 = n // 3, 2 * n // 3

    def third_rate(start, end):
        ts = tap_times[start:end]
        if len(ts) < 2:
            return np.nan
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def third_amp(start, end):
        a = tap_amps[start:end]
        return float(np.nanmean(a)) if len(a) > 0 else np.nan

    early_rate = third_rate(0, t1)
    mid_rate   = third_rate(t1, t2)
    late_rate  = third_rate(t2, n)
    early_amp  = third_amp(0, t1)
    mid_amp    = third_amp(t1, t2)
    late_amp   = third_amp(t2, n)

    def decrement(early, late):
        if np.isnan(early) or np.isnan(late) or early < 1e-6:
            return np.nan
        return float((early - late) / early * 100)

    rate_dec = decrement(early_rate, late_rate)
    amp_dec  = decrement(early_amp, late_amp)

    flags = _generate_flags(
        mean_rate, rhythm_cv, rate_dec, amp_dec,
        int(arrest_mask.sum()), arrest_times,
    )

    return TapAnalysis(
        hand=hand_label, fps=fps, duration=frame_count / fps,
        detection_rate=detection_rate,
        time=t, distance=dist_smooth,
        tap_times=tap_times, tap_amplitudes=tap_amps,
        mean_rate_hz=float(mean_rate),
        mean_amplitude=float(np.nanmean(tap_amps)),
        rhythm_cv=float(rhythm_cv),
        early_rate_hz=early_rate, mid_rate_hz=mid_rate, late_rate_hz=late_rate,
        early_amplitude=early_amp, mid_amplitude=mid_amp, late_amplitude=late_amp,
        rate_decrement_pct=rate_dec, amplitude_decrement_pct=amp_dec,
        arrest_count=int(arrest_mask.sum()), arrest_times=arrest_times,
        deviation_flags=flags,
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
    if total / fps < 5:
        print(f"  Warning: short video ({total/fps:.1f}s) — recommend 10s for tap test")

    print(f"  Mode: tap  |  {fps:.1f} fps  |  ~{total/fps:.0f}s  ({total} frames)")

    distances  = []
    ref_sizes  = []
    hand_votes = {"Left": 0, "Right": 0}
    n = detected = 0

    cap = cv2.VideoCapture(video_path)
    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            h, w = frame.shape[:2]
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_hand_landmarks and result.multi_handedness:
                detected += 1
                lms   = result.multi_hand_landmarks[0].landmark
                label = result.multi_handedness[0].classification[0].label
                hand_votes[label] = hand_votes.get(label, 0) + 1

                thumb = np.array([lms[THUMB_TIP].x * w, lms[THUMB_TIP].y * h])
                index = np.array([lms[INDEX_TIP].x * w, lms[INDEX_TIP].y * h])
                distances.append(np.linalg.norm(thumb - index))

                a = np.array([lms[HAND_REF[0]].x * w, lms[HAND_REF[0]].y * h])
                b = np.array([lms[HAND_REF[1]].x * w, lms[HAND_REF[1]].y * h])
                ref_sizes.append(np.linalg.norm(b - a))
            else:
                distances.append(np.nan)
                ref_sizes.append(np.nan)

            if n % max(1, int(fps)) == 0:
                print(f"\r  Tracking... {n/fps:.0f}s", end="", flush=True)

    cap.release()
    rate      = detected / n if n > 0 else 0.0
    hand_label = max(hand_votes, key=hand_votes.get) if hand_votes else "Unknown"
    ref_size   = float(np.nanmedian(ref_sizes)) if ref_sizes else 100.0
    print(f"\r  Tracked {n/fps:.0f}s — hand detected in {rate*100:.0f}% of frames ({hand_label})")

    return np.array(distances), ref_size, hand_label, fps, n, rate


# ---------------------------------------------------------------------------
# Tap detection
# ---------------------------------------------------------------------------

def _detect_taps(dist_smooth, fps):
    min_dist = max(3, int(fps * 0.12))  # 120 ms minimum between taps

    # Taps = local minima (fingers close)
    troughs, _ = signal.find_peaks(-dist_smooth, distance=min_dist, prominence=0.02)
    # Peaks = local maxima (fingers apart)
    peaks, _   = signal.find_peaks(dist_smooth,  distance=min_dist, prominence=0.02)

    # Amplitude of each tap = preceding peak height − trough depth
    amplitudes = []
    for trough_i in troughs:
        pre_peaks = peaks[peaks < trough_i]
        if len(pre_peaks) > 0:
            p = pre_peaks[-1]
            amplitudes.append(max(0.0, float(dist_smooth[p] - dist_smooth[trough_i])))
        else:
            amplitudes.append(float(dist_smooth[:trough_i].max() - dist_smooth[trough_i])
                              if trough_i > 0 else np.nan)

    intervals = np.diff(troughs) / fps if len(troughs) > 1 else np.array([])
    return troughs, amplitudes, intervals


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def _generate_flags(mean_rate, rhythm_cv, rate_dec, amp_dec, arrest_count, arrest_times):
    flags = []

    if not np.isnan(mean_rate) and mean_rate < RATE_LOW_THRESH:
        flags.append(
            f"SLOW TAP RATE: {mean_rate:.2f} Hz — below threshold {RATE_LOW_THRESH} Hz. "
            "Normal is ~4–5 Hz. Bradykinesia likely."
        )

    if not np.isnan(amp_dec) and amp_dec > AMPLITUDE_DECREMENT_THRESH:
        flags.append(
            f"AMPLITUDE DECREMENT: {amp_dec:.1f}% reduction (early → late) — "
            f"threshold {AMPLITUDE_DECREMENT_THRESH}%. Characteristic of PD bradykinesia."
        )

    if not np.isnan(rate_dec) and rate_dec > RATE_DECREMENT_THRESH:
        flags.append(
            f"RATE DECREMENT: {rate_dec:.1f}% slowing (early → late) — "
            f"threshold {RATE_DECREMENT_THRESH}%."
        )

    if arrest_count > 0:
        flags.append(
            f"ARRESTS DETECTED: {arrest_count} pause(s) exceeding "
            f"{ARREST_MULTIPLIER}× the median interval. "
            "Arrests are a UPDRS bradykinesia marker."
        )

    if not np.isnan(rhythm_cv) and rhythm_cv > RHYTHM_CV_THRESH:
        flags.append(
            f"IRREGULAR TAPPING: rhythm CV {rhythm_cv:.1f}% — "
            f"threshold {RHYTHM_CV_THRESH}%."
        )

    return flags


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_tap_report(analysis: TapAnalysis, video_path: str):
    print()
    print("=" * 64)
    print("  FINGER TAPPING REPORT")
    print("=" * 64)
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Hand     : {analysis.hand}")
    print(f"  Duration : {analysis.duration:.1f} s  |  {analysis.fps:.1f} fps  |  "
          f"detection {analysis.detection_rate*100:.0f}%")
    print(f"  Taps     : {len(analysis.tap_times)}")
    print()
    print(f"  OVERALL")
    print(f"    Tap rate    : {analysis.mean_rate_hz:.2f} Hz  ({analysis.mean_rate_hz*60:.0f}/min)")
    print(f"    Amplitude   : {analysis.mean_amplitude:.3f} × hand width")
    print(f"    Rhythm CV   : {analysis.rhythm_cv:.1f}%")
    print(f"    Arrests     : {analysis.arrest_count}")
    print()
    print(f"  DECREMENT  (early → mid → late third)")

    def fmt(v): return f"{v:.2f} Hz" if not np.isnan(v) else "N/A"
    def fmta(v): return f"{v:.3f}" if not np.isnan(v) else "N/A"
    def fmtd(v): return f"{v:+.1f}%" if not np.isnan(v) else "N/A"

    print(f"    Rate        : {fmt(analysis.early_rate_hz)} → {fmt(analysis.mid_rate_hz)} → "
          f"{fmt(analysis.late_rate_hz)}   (decrement {fmtd(analysis.rate_decrement_pct)})")
    print(f"    Amplitude   : {fmta(analysis.early_amplitude)} → {fmta(analysis.mid_amplitude)} → "
          f"{fmta(analysis.late_amplitude)}   (decrement {fmtd(analysis.amplitude_decrement_pct)})")
    print()

    if analysis.deviation_flags:
        print("  FLAGS / DEVIATIONS:")
        for flag in analysis.deviation_flags:
            print(f"    !! {flag}")
    else:
        print("  No deviations detected.")

    print("=" * 64)
    print()


def save_tap_outputs(analysis: TapAnalysis, output_dir, video_path: str):
    stem = Path(video_path).stem
    out  = Path(output_dir) if output_dir else Path(video_path).parent / f"{stem}_tap"
    out.mkdir(parents=True, exist_ok=True)

    plot_path = _save_plot(analysis, out, stem, video_path)
    json_path = _save_json(analysis, out, stem, video_path)
    print(f"  Plot : {plot_path}")
    print(f"  JSON : {json_path}")


def _save_plot(analysis, out_dir, stem, video_path):
    t = analysis.time

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Finger Tapping — {Path(video_path).name}  ({analysis.hand} hand)\n"
        f"{len(analysis.tap_times)} taps  |  {analysis.mean_rate_hz:.2f} Hz mean  |  "
        f"amp decrement {analysis.amplitude_decrement_pct:+.1f}%  |  "
        f"{analysis.arrest_count} arrest(s)",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── Row 0: Distance time series with tap markers ─────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, analysis.distance, lw=0.8, color="steelblue", alpha=0.9)
    for tt in analysis.tap_times:
        ax0.axvline(tt, color="limegreen", lw=0.6, alpha=0.5)
    for at in analysis.arrest_times:
        ax0.axvline(at, color="crimson", lw=1.5, alpha=0.7, ls="--")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Thumb–index distance\n(normalized)")
    ax0.set_title("Thumb–Index Distance  (green = tap  |  red dashed = arrest)")
    ax0.set_xlim(t[0], t[-1])

    # ── Row 1 left: Tap rate over time ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    if len(analysis.tap_times) >= 3:
        ivs   = np.diff(analysis.tap_times)
        rates = 1.0 / ivs
        ax1.plot(analysis.tap_times[1:], rates, "o-", color="steelblue",
                 lw=1.2, ms=4, alpha=0.85)
        ax1.axhline(RATE_LOW_THRESH, color="crimson", ls="--", lw=1,
                    label=f"Low threshold ({RATE_LOW_THRESH} Hz)")
        ax1.axhline(analysis.mean_rate_hz, color="gray", ls=":", lw=1,
                    label=f"Mean {analysis.mean_rate_hz:.2f} Hz")
        ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Instantaneous rate (Hz)")
    ax1.set_title("Tap Rate Over Time")

    # ── Row 1 right: Tap amplitude over time ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    if len(analysis.tap_times) > 0 and len(analysis.tap_amplitudes) > 0:
        ax2.plot(analysis.tap_times[:len(analysis.tap_amplitudes)],
                 analysis.tap_amplitudes, "o-", color="darkorange",
                 lw=1.2, ms=4, alpha=0.85)
        ax2.axhline(analysis.mean_amplitude, color="gray", ls=":", lw=1,
                    label=f"Mean {analysis.mean_amplitude:.3f}")
        ax2.legend(fontsize=8)
        ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (normalized)")
    ax2.set_title("Tap Amplitude Over Time")

    # ── Row 2 left: Rate by third ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    thirds = ["Early", "Mid", "Late"]
    rates  = [analysis.early_rate_hz, analysis.mid_rate_hz, analysis.late_rate_hz]
    colors = []
    for r in rates:
        colors.append("steelblue" if np.isnan(r) or r >= RATE_LOW_THRESH else "crimson")
    bars = ax3.bar(thirds, [r if not np.isnan(r) else 0 for r in rates],
                   color=colors, edgecolor="black", lw=0.5)
    ax3.axhline(RATE_LOW_THRESH, color="crimson", ls="--", lw=1,
                label=f"Low threshold ({RATE_LOW_THRESH} Hz)")
    for bar, val in zip(bars, rates):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Tap rate (Hz)")
    ax3.set_title(f"Rate by Third  (decrement {analysis.rate_decrement_pct:+.1f}%)"
                  if not np.isnan(analysis.rate_decrement_pct) else "Rate by Third")
    ax3.legend(fontsize=8)

    # ── Row 2 right: Amplitude by third ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    amps   = [analysis.early_amplitude, analysis.mid_amplitude, analysis.late_amplitude]
    colors = ["darkorange"] * 3
    bars = ax4.bar(thirds, [a if not np.isnan(a) else 0 for a in amps],
                   color=colors, edgecolor="black", lw=0.5)
    for bar, val in zip(bars, amps):
        if not np.isnan(val):
            ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax4.set_ylabel("Amplitude (normalized)")
    ax4.set_title(f"Amplitude by Third  (decrement {analysis.amplitude_decrement_pct:+.1f}%)"
                  if not np.isnan(analysis.amplitude_decrement_pct) else "Amplitude by Third")

    plot_path = out_dir / f"{stem}_tap.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_json(analysis, out_dir, stem, video_path):
    def f(v): return round(float(v), 4) if not np.isnan(v) else None

    summary = {
        "video":               Path(video_path).name,
        "mode":                "tap",
        "hand":                analysis.hand,
        "duration_s":          round(analysis.duration, 2),
        "fps":                 round(analysis.fps, 2),
        "detection_rate_pct":  round(analysis.detection_rate * 100, 1),
        "tap_count":           len(analysis.tap_times),
        "mean_rate_hz":        f(analysis.mean_rate_hz),
        "mean_amplitude":      f(analysis.mean_amplitude),
        "rhythm_cv_pct":       f(analysis.rhythm_cv),
        "arrest_count":        analysis.arrest_count,
        "rate_by_third": {
            "early_hz": f(analysis.early_rate_hz),
            "mid_hz":   f(analysis.mid_rate_hz),
            "late_hz":  f(analysis.late_rate_hz),
            "decrement_pct": f(analysis.rate_decrement_pct),
        },
        "amplitude_by_third": {
            "early": f(analysis.early_amplitude),
            "mid":   f(analysis.mid_amplitude),
            "late":  f(analysis.late_amplitude),
            "decrement_pct": f(analysis.amplitude_decrement_pct),
        },
        "flags":     analysis.deviation_flags,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = out_dir / f"{stem}_tap.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return json_path
