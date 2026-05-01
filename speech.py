import cv2
import json
import subprocess
import tempfile
import os
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from scipy import signal

mp_face = mp.solutions.face_mesh

# ── Audio parameters ──────────────────────────────────────────────────────────
SR        = 22050
HOP       = 512      # ~23 ms per frame
FRAME_LEN = 2048

# ── Clinical parameters ───────────────────────────────────────────────────────
VOCAL_TREMOR_BAND = (3.0, 8.0)   # Hz — PD vocal tremor
PAUSE_MIN_S       = 0.25
LONG_PAUSE_S      = 2.0

# ── Hypomimia landmark groups (MediaPipe Face Mesh indices) ───────────────────
LIP_IDX   = [61, 291, 13, 14, 78, 308]
BROW_IDX  = [70, 63, 105, 107, 336, 296, 334, 300]
JAW_IDX   = [152, 172, 397]
LEFT_IDX  = [61, 70, 105, 172]
RIGHT_IDX = [291, 336, 334, 397]
EYE_REF   = (33, 263)

# ── Flag thresholds ───────────────────────────────────────────────────────────
LOUDNESS_LOW_DB     = -33.0
PITCH_RANGE_LOW_HZ  = 30.0
TREMOR_POWER_THRESH = 0.25
MOBILITY_LOW        = 0.008
LR_ASYMMETRY_THRESH = 1.8


@dataclass
class VoiceResult:
    duration_s: float
    mean_loudness_db: float
    pitch_mean_hz: float
    pitch_range_hz: float
    pitch_cv: float
    vocal_tremor_freq_hz: float
    vocal_tremor_power: float
    voiced_fraction: float
    pause_count: int
    mean_pause_s: float
    long_pause_count: int


@dataclass
class HypomimiaResult:
    overall_mobility: float
    lip_mobility: float
    brow_mobility: float
    jaw_mobility: float
    left_mobility: float
    right_mobility: float
    lr_asymmetry: float
    detection_rate: float


@dataclass
class SpeechAnalysis:
    duration: float
    voice: VoiceResult
    face: object                  # HypomimiaResult or None
    deviation_flags: list
    # Time series stored for plotting
    rms_db: np.ndarray            # amplitude envelope in dB
    rms_times: np.ndarray         # time axis for rms_db
    f0: np.ndarray                # pitch track (NaN where unvoiced)
    pitch_times: np.ndarray       # time axis for f0
    voiced_flag: np.ndarray       # bool array aligned with rms_times
    tremor_freqs: np.ndarray      # Hz axis for tremor spectrum
    tremor_power: np.ndarray      # power spectrum of amplitude envelope


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_speech(video_path: str) -> SpeechAnalysis:
    print("  Extracting audio...")
    y, sr = _extract_audio(video_path)

    print("  Analyzing voice...")
    voice, arrays = _analyze_voice(y, sr)

    print("  Tracking face...")
    face = _analyze_face(video_path)

    return SpeechAnalysis(
        duration=voice.duration_s,
        voice=voice,
        face=face,
        deviation_flags=_generate_flags(voice, face),
        **arrays,
    )


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-ac", "1", "-ar", str(SR),
             "-vn", tmp_path, "-y", "-loglevel", "error"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
        y, sr = librosa.load(tmp_path, sr=SR, mono=True)
    except FileNotFoundError:
        raise ValueError("ffmpeg not found — install with: brew install ffmpeg")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if len(y) < sr * 2:
        raise ValueError("Audio too short — need at least 2 seconds of speech.")

    return y, sr


# ---------------------------------------------------------------------------
# Voice analysis
# ---------------------------------------------------------------------------

def _analyze_voice(y, sr):
    """Returns (VoiceResult, dict_of_arrays_for_plotting)."""
    duration_s  = len(y) / sr
    frame_rate  = sr / HOP

    # Amplitude envelope
    rms      = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP)[0]
    rms_db   = librosa.amplitude_to_db(rms, ref=1.0)
    rms_t    = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    mean_ldb = float(np.mean(rms_db))

    # Pitch via pyin
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=65.0, fmax=500.0, sr=sr, hop_length=HOP, fill_na=np.nan,
    )
    pitch_t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)

    # Align voiced_flag to rms length (pyin sometimes differs by 1)
    n = min(len(rms), len(voiced_flag))
    voiced_flag = voiced_flag[:n]
    rms_db_n    = rms_db[:n]
    rms_t_n     = rms_t[:n]

    f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([])
    if len(f0_voiced) >= 10:
        pitch_mean  = float(np.nanmean(f0_voiced))
        pitch_range = float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced))
        pitch_cv    = float(np.nanstd(f0_voiced) / pitch_mean * 100) if pitch_mean > 0 else np.nan
    else:
        pitch_mean = pitch_range = pitch_cv = np.nan

    voiced_fraction = float(voiced_flag.mean())

    # Pauses
    pauses = _detect_pauses(voiced_flag, frame_rate)

    # Vocal tremor: FFT of smoothed amplitude envelope
    rms_smooth = np.convolve(rms[:n], np.ones(5) / 5, mode="same")
    tfreqs  = np.fft.rfftfreq(len(rms_smooth), 1.0 / frame_rate)
    tpower  = np.abs(np.fft.rfft(rms_smooth)) ** 2
    t_mask  = (tfreqs >= VOCAL_TREMOR_BAND[0]) & (tfreqs <= VOCAL_TREMOR_BAND[1])

    if t_mask.any() and tpower.sum() > 0:
        tremor_freq  = float(tfreqs[t_mask][np.argmax(tpower[t_mask])])
        tremor_power = float(tpower[t_mask].sum() / tpower.sum())
    else:
        tremor_freq = tremor_power = np.nan

    result = VoiceResult(
        duration_s=duration_s,
        mean_loudness_db=mean_ldb,
        pitch_mean_hz=pitch_mean,
        pitch_range_hz=pitch_range,
        pitch_cv=pitch_cv,
        vocal_tremor_freq_hz=tremor_freq,
        vocal_tremor_power=tremor_power,
        voiced_fraction=voiced_fraction,
        pause_count=len(pauses),
        mean_pause_s=float(np.mean(pauses)) if pauses else 0.0,
        long_pause_count=int(sum(p > LONG_PAUSE_S for p in pauses)),
    )

    arrays = dict(
        rms_db=rms_db_n,
        rms_times=rms_t_n,
        f0=f0[:n],
        pitch_times=pitch_t[:n],
        voiced_flag=voiced_flag,
        tremor_freqs=tfreqs,
        tremor_power=tpower,
    )

    return result, arrays


def _detect_pauses(voiced_flag, frame_rate):
    min_frames = int(PAUSE_MIN_S * frame_rate)
    pauses, count = [], 0
    for v in voiced_flag:
        if not v:
            count += 1
        else:
            if count >= min_frames:
                pauses.append(count / frame_rate)
            count = 0
    if count >= min_frames:
        pauses.append(count / frame_rate)
    return pauses


# ---------------------------------------------------------------------------
# Hypomimia analysis
# ---------------------------------------------------------------------------

def _analyze_face(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    n = detected = 0
    frames_lm = []

    with mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            h, w = frame.shape[:2]
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_face_landmarks:
                detected += 1
                lms = result.multi_face_landmarks[0].landmark
                frames_lm.append(np.array([[lm.x * w, lm.y * h] for lm in lms]))
            else:
                frames_lm.append(None)
            if n % 30 == 0:
                print(f"\r  Face tracking... {n/30:.0f}s", end="", flush=True)

    cap.release()
    print()

    valid = [x for x in frames_lm if x is not None]
    if len(valid) < 15:
        print("  Warning: face not reliably detected — hypomimia skipped")
        return None

    lm = np.array(valid)
    ref = float(np.mean(
        np.linalg.norm(lm[:, EYE_REF[0]] - lm[:, EYE_REF[1]], axis=1)
    ))

    std_x = lm[:, :, 0].std(axis=0)
    std_y = lm[:, :, 1].std(axis=0)
    mob   = np.sqrt(std_x ** 2 + std_y ** 2) / ref

    def gm(idx): return float(mob[idx].mean())

    left  = gm(LEFT_IDX)
    right = gm(RIGHT_IDX)
    lr    = max(left, right) / min(left, right) if min(left, right) > 1e-6 else np.nan

    return HypomimiaResult(
        overall_mobility=float(mob.mean()),
        lip_mobility=gm(LIP_IDX),
        brow_mobility=gm(BROW_IDX),
        jaw_mobility=gm(JAW_IDX),
        left_mobility=left,
        right_mobility=right,
        lr_asymmetry=lr,
        detection_rate=detected / n if n > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def _generate_flags(voice: VoiceResult, face) -> list:
    flags = []
    v = voice

    if v.mean_loudness_db < LOUDNESS_LOW_DB:
        flags.append(
            f"HYPOPHONIA: mean loudness {v.mean_loudness_db:.1f} dB — "
            f"below {LOUDNESS_LOW_DB} dB. Reduced vocal volume is common in PD. "
            "Note: compressed video (e.g. Zoom) limits absolute loudness comparisons."
        )
    if not np.isnan(v.pitch_range_hz) and v.pitch_range_hz < PITCH_RANGE_LOW_HZ:
        flags.append(
            f"MONOTONE SPEECH: pitch range {v.pitch_range_hz:.0f} Hz — "
            f"below {PITCH_RANGE_LOW_HZ} Hz. Reduced prosodic variation occurs in PD and AD."
        )
    if not np.isnan(v.vocal_tremor_power) and v.vocal_tremor_power > TREMOR_POWER_THRESH:
        flags.append(
            f"VOCAL TREMOR: {v.vocal_tremor_power*100:.0f}% of voice amplitude power in "
            f"{VOCAL_TREMOR_BAND[0]}–{VOCAL_TREMOR_BAND[1]} Hz "
            f"(dominant {v.vocal_tremor_freq_hz:.1f} Hz)."
        )
    if v.long_pause_count > 0:
        flags.append(
            f"LONG PAUSES: {v.long_pause_count} pause(s) > {LONG_PAUSE_S:.0f}s. "
            "Occurs in PD (initiation difficulty) and AD (word-finding)."
        )
    if face is not None:
        if face.overall_mobility < MOBILITY_LOW:
            flags.append(
                f"HYPOMIMIA: facial mobility {face.overall_mobility:.4f} — "
                f"below {MOBILITY_LOW}. Masked face is a recognized PD sign."
            )
        if not np.isnan(face.lr_asymmetry) and face.lr_asymmetry > LR_ASYMMETRY_THRESH:
            side = "right" if face.right_mobility < face.left_mobility else "left"
            flags.append(
                f"ASYMMETRIC HYPOMIMIA: {face.lr_asymmetry:.2f}× "
                f"({side} face less mobile) — consistent with asymmetric PD onset."
            )
    return flags


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_speech_report(analysis: SpeechAnalysis, video_path: str):
    v = analysis.voice
    f = analysis.face

    print()
    print("=" * 64)
    print("  SPEECH & FACIAL EXPRESSION REPORT")
    print("=" * 64)
    print(f"  Video    : {Path(video_path).name}")
    print(f"  Duration : {analysis.duration:.1f} s")
    print()
    print("  VOICE")
    print(f"    Loudness      : {v.mean_loudness_db:.1f} dB (mean RMS)")
    if not np.isnan(v.pitch_mean_hz):
        print(f"    Pitch         : {v.pitch_mean_hz:.0f} Hz mean  |  "
              f"range {v.pitch_range_hz:.0f} Hz  |  CV {v.pitch_cv:.1f}%")
    if not np.isnan(v.vocal_tremor_power):
        print(f"    Vocal tremor  : {v.vocal_tremor_power*100:.0f}% power in "
              f"{VOCAL_TREMOR_BAND[0]}–{VOCAL_TREMOR_BAND[1]} Hz  "
              f"(dominant {v.vocal_tremor_freq_hz:.1f} Hz)")
    print(f"    Speech        : {v.voiced_fraction*100:.0f}% voiced  |  "
          f"{v.pause_count} pause(s)  |  {v.long_pause_count} long (>{LONG_PAUSE_S:.0f}s)")
    print()

    if f is not None:
        print("  FACIAL EXPRESSION")
        print(f"    Detection     : {f.detection_rate*100:.0f}% of frames")
        print(f"    Overall mobility  : {f.overall_mobility:.4f}")
        print(f"    Lip / Brow / Jaw  : {f.lip_mobility:.4f}  /  "
              f"{f.brow_mobility:.4f}  /  {f.jaw_mobility:.4f}")
        print(f"    Left / Right      : {f.left_mobility:.4f}  /  "
              f"{f.right_mobility:.4f}  (asymmetry {f.lr_asymmetry:.2f}×)")
        print()

    if analysis.deviation_flags:
        print("  FLAGS / DEVIATIONS:")
        for flag in analysis.deviation_flags:
            print(f"    !! {flag}")
    else:
        print("  No deviations detected.")

    print("=" * 64)
    print()


def save_speech_outputs(analysis: SpeechAnalysis, output_dir, video_path: str):
    stem = Path(video_path).stem
    out  = Path(output_dir) if output_dir else Path(video_path).parent / f"{stem}_speech"
    out.mkdir(parents=True, exist_ok=True)
    plot_path = _save_plot(analysis, out, stem, video_path)
    json_path = _save_json(analysis, out, stem, video_path)
    print(f"  Plot : {plot_path}")
    print(f"  JSON : {json_path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _save_plot(analysis, out_dir, stem, video_path):
    v  = analysis.voice
    fc = analysis.face

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Speech & Facial Expression — {Path(video_path).name}\n"
        f"Duration: {v.duration_s:.1f}s  |  "
        f"Loudness: {v.mean_loudness_db:.1f} dB  |  "
        f"Pitch range: {v.pitch_range_hz:.0f} Hz  |  "
        f"Vocal tremor: {v.vocal_tremor_power*100:.0f}%",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── Row 0: Amplitude envelope + voiced/unvoiced ───────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(analysis.rms_times, analysis.rms_db, lw=0.8, color="steelblue", alpha=0.9)
    # Shade unvoiced regions
    unv = ~analysis.voiced_flag
    for i in range(len(unv)):
        if unv[i]:
            ax0.axvspan(analysis.rms_times[i],
                        analysis.rms_times[min(i + 1, len(analysis.rms_times) - 1)],
                        color="lightgray", alpha=0.4, lw=0)
    ax0.axhline(LOUDNESS_LOW_DB, color="crimson", lw=1, ls="--",
                label=f"Hypophonia threshold ({LOUDNESS_LOW_DB} dB)")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Loudness (dB RMS)")
    ax0.set_title("Amplitude Envelope  (gray = silence/unvoiced)")
    ax0.set_xlim(analysis.rms_times[0], analysis.rms_times[-1])
    ax0.legend(fontsize=8)

    # ── Row 1 left: Pitch over time ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    voiced_f0 = analysis.f0.copy()
    voiced_f0[~analysis.voiced_flag] = np.nan
    ax1.plot(analysis.pitch_times, voiced_f0, lw=0.9, color="darkorange", alpha=0.85)
    if not np.isnan(v.pitch_mean_hz):
        ax1.axhline(v.pitch_mean_hz, color="gray", lw=1, ls=":",
                    label=f"Mean {v.pitch_mean_hz:.0f} Hz")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Fundamental freq (Hz)")
    ax1.set_title(f"Pitch (F0)  — range {v.pitch_range_hz:.0f} Hz"
                  if not np.isnan(v.pitch_range_hz) else "Pitch (F0)")
    ax1.set_xlim(analysis.rms_times[0], analysis.rms_times[-1])
    ax1.legend(fontsize=8)

    # ── Row 1 right: Vocal tremor spectrum ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    fmask = (analysis.tremor_freqs >= 0.5) & (analysis.tremor_freqs <= 12)
    ax2.semilogy(analysis.tremor_freqs[fmask], analysis.tremor_power[fmask],
                 color="steelblue", lw=1.2)
    ax2.axvspan(*VOCAL_TREMOR_BAND, alpha=0.15, color="crimson",
                label=f"Tremor band ({VOCAL_TREMOR_BAND[0]}–{VOCAL_TREMOR_BAND[1]} Hz)")
    if not np.isnan(v.vocal_tremor_freq_hz):
        ax2.axvline(v.vocal_tremor_freq_hz, color="crimson", lw=1.5, ls="--",
                    label=f"Peak {v.vocal_tremor_freq_hz:.1f} Hz")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (log)")
    ax2.set_title(f"Voice Amplitude Spectrum  ({v.vocal_tremor_power*100:.0f}% in tremor band)")
    ax2.legend(fontsize=8)

    # ── Row 2 left: Facial mobility by region ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    if fc is not None:
        regions = ["Overall", "Lips", "Brow", "Jaw"]
        vals    = [fc.overall_mobility, fc.lip_mobility, fc.brow_mobility, fc.jaw_mobility]
        colors  = ["crimson" if val < MOBILITY_LOW else "steelblue" for val in vals]
        bars = ax3.bar(regions, vals, color=colors, edgecolor="black", lw=0.5)
        ax3.axhline(MOBILITY_LOW, color="crimson", ls="--", lw=1,
                    label=f"Low threshold ({MOBILITY_LOW})")
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.0002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax3.set_ylabel("Mobility (normalized to face size)")
        ax3.set_title("Facial Mobility by Region")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Face not detected", ha="center", va="center",
                 transform=ax3.transAxes, color="gray", fontsize=12)
        ax3.axis("off")
        ax3.set_title("Facial Mobility")

    # ── Row 2 right: Left vs right ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    if fc is not None:
        vals   = [fc.left_mobility, fc.right_mobility]
        bars = ax4.bar(["Left", "Right"], vals,
                       color=["steelblue", "darkorange"], edgecolor="black", lw=0.5)
        for bar, val in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.0002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        asym = f"{fc.lr_asymmetry:.2f}×" if not np.isnan(fc.lr_asymmetry) else "N/A"
        ax4.set_title(f"Left vs Right Mobility  (asymmetry {asym})")
        ax4.set_ylabel("Mobility (normalized)")
        if not np.isnan(fc.lr_asymmetry) and fc.lr_asymmetry > LR_ASYMMETRY_THRESH:
            ax4.set_facecolor("#fff0f0")
    else:
        ax4.text(0.5, 0.5, "Face not detected", ha="center", va="center",
                 transform=ax4.transAxes, color="gray", fontsize=12)
        ax4.axis("off")
        ax4.set_title("Left vs Right Facial Mobility")

    plot_path = out_dir / f"{stem}_speech.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _save_json(analysis, out_dir, stem, video_path):
    def f(v):
        if v is None: return None
        try:
            return None if np.isnan(float(v)) else round(float(v), 4)
        except Exception:
            return None

    vr = analysis.voice
    fc = analysis.face

    summary = {
        "video":      Path(video_path).name,
        "mode":       "speech",
        "duration_s": round(analysis.duration, 2),
        "voice": {
            "mean_loudness_db":       f(vr.mean_loudness_db),
            "pitch_mean_hz":          f(vr.pitch_mean_hz),
            "pitch_range_hz":         f(vr.pitch_range_hz),
            "pitch_cv_pct":           f(vr.pitch_cv),
            "vocal_tremor_freq_hz":   f(vr.vocal_tremor_freq_hz),
            "vocal_tremor_power_pct": f(vr.vocal_tremor_power * 100)
                                      if not np.isnan(vr.vocal_tremor_power) else None,
            "voiced_fraction":        f(vr.voiced_fraction),
            "pause_count":            vr.pause_count,
            "mean_pause_s":           f(vr.mean_pause_s),
            "long_pause_count":       vr.long_pause_count,
        },
        "face": {
            "detection_rate_pct": f(fc.detection_rate * 100),
            "overall_mobility":   f(fc.overall_mobility),
            "lip_mobility":       f(fc.lip_mobility),
            "brow_mobility":      f(fc.brow_mobility),
            "jaw_mobility":       f(fc.jaw_mobility),
            "left_mobility":      f(fc.left_mobility),
            "right_mobility":     f(fc.right_mobility),
            "lr_asymmetry":       f(fc.lr_asymmetry),
        } if fc is not None else None,
        "flags":     analysis.deviation_flags,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = out_dir / f"{stem}_speech.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return json_path
