#!/usr/bin/env python3
"""
Longitudinal batch processor — tremor, gait, tap, and speech analysis.

Scans a directory (recursively) for patient videos, runs the appropriate
analysis on each, organizes outputs by patient, and generates longitudinal
comparison reports showing improvement or worsening from baseline.

Usage:
  python batch.py VIDEO_DIR [--output-dir DIR] [--manifest CSV]
                            [--resume] [--dry-run]

Video discovery (no manifest needed):
  Recursively finds all .mp4 / .mov / .avi / .mkv files and attempts to
  parse patient ID, date, and mode from the filename or directory structure.

  Preferred filename pattern:
    PATIENTID_YYYYMMDD_MODE.mp4       e.g.  PT001_20240115_hands.mp4
    PATIENTID_YYYY-MM-DD_MODE.mp4     e.g.  PT001_2024-01-15_gait.mp4

  If the filename doesn't match, the parent directory name is used as the
  patient ID, the date is parsed from wherever it appears, and mode
  defaults to 'auto'. File modification time is used as a last resort for
  the date.

  Recognized modes: hands  feet  face  gait  tap  speech  auto

Manifest CSV (optional — overrides filename parsing):
  patient_id,date,mode,path
  PT001,2024-01-15,hands,PT001/jan_hands.mp4
  Paths may be absolute or relative to the manifest file location.

Output structure:
  OUTPUT_DIR/
  ├── PT001/
  │   ├── 20240115_baseline_<stem>/   one folder per session
  │   │   ├── <stem>_tremor.json
  │   │   └── <stem>_tremor.png
  │   ├── 20240415_wk12_<stem>/
  │   ├── longitudinal.png            metric trends over time
  │   └── longitudinal_summary.json
  └── batch_summary.csv               all patients side-by-side
"""

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXT   = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
KNOWN_MODES = {"hands", "feet", "face", "gait", "tap", "speech", "auto"}

# Regex: PATIENTID_DATE_MODE (with optional extra suffix)
_FNAME_RE = re.compile(
    r"^(.+?)_(\d{4}-?\d{2}-?\d{2})_(" + "|".join(KNOWN_MODES) + r")(?:_.+)?$",
    re.IGNORECASE,
)

# For each tracked metric: (display label, unit, better_direction)
# better_direction: "lower", "higher", or None (ambiguous / informational)
METRIC_META = {
    "tremor_amplitude":      ("Tremor amplitude",       "norm",    "lower"),
    "tremor_freq_hz":        ("Tremor frequency",        "Hz",      None),
    "pd_band_power_pct":     ("PD-band power",           "%",       None),
    "gait_cadence_spm":      ("Gait cadence",            "spm",     None),
    "gait_step_cv_pct":      ("Step irregularity (CV)",  "%",       "lower"),
    "gait_arm_asymmetry":    ("Arm swing asymmetry",     "×",       "lower"),
    "tap_rate_hz":           ("Tap rate",                "Hz",      "higher"),
    "tap_rhythm_cv_pct":     ("Tap rhythm CV",           "%",       "lower"),
    "tap_arrests":           ("Tap arrests",             "count",   "lower"),
    "tap_rate_decrement":    ("Rate decrement",          "%",       "lower"),
    "tap_amp_decrement":     ("Amplitude decrement",     "%",       "lower"),
    "voice_loudness_db":     ("Voice loudness",          "dB",      "higher"),
    "voice_pitch_range_hz":  ("Pitch range",             "Hz",      "higher"),
    "voice_tremor_pct":      ("Vocal tremor power",      "%",       "lower"),
    "voice_long_pauses":     ("Long pauses",             "count",   "lower"),
    "face_mobility":         ("Facial mobility",         "norm",    "higher"),
    "face_asymmetry":        ("Facial asymmetry",        "×",       "lower"),
}

CHANGE_THRESHOLD_PCT = 10.0   # % change required to call improved / worsened


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class VideoRecord:
    patient_id: str
    date: date
    mode: str
    path: Path
    session_label: str = ""
    roi: str = ""          # optional 'x,y,w,h' for --mode flow


@dataclass
class SessionResult:
    record: VideoRecord
    session_dir: Path
    metrics: dict
    success: bool
    error: str = ""


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def scan(video_dir: Path, manifest: Path = None) -> list:
    if manifest:
        return _from_manifest(manifest)
    return _from_directory(video_dir)


def _from_manifest(manifest_path: Path) -> list:
    records = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            path = Path(row["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            dt = _parse_date(row["date"])
            if dt is None:
                print(f"  Warning: bad date '{row['date']}' in manifest — skipping")
                continue
            records.append(VideoRecord(
                patient_id=row["patient_id"].strip(),
                date=dt,
                mode=row["mode"].strip().lower(),
                path=path,
                roi=row.get("roi", "").strip(),
            ))
    return records


def _from_directory(root: Path) -> list:
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXT:
            continue
        rec = _parse_path(path, root)
        if rec:
            records.append(rec)
        else:
            print(f"  Skipping (unparseable): {path.relative_to(root)}")
    return records


def _parse_path(path: Path, root: Path) -> VideoRecord:
    stem  = path.stem
    parts = path.relative_to(root).parts

    # Try full PATIENTID_DATE_MODE pattern
    m = _FNAME_RE.match(stem)
    if m:
        dt = _parse_date(m.group(2))
        if dt:
            return VideoRecord(
                patient_id=m.group(1),
                date=dt,
                mode=m.group(3).lower(),
                path=path,
            )

    # Fall back: top-level subdirectory = patient ID
    patient_id = parts[0] if len(parts) > 1 else stem

    # Search the full relative path for a date (covers dates in any subdirectory)
    full_rel = str(path.relative_to(root))
    dt   = _date_from_text(full_rel)
    mode = _mode_from_text(stem) or _mode_from_text(full_rel) or "auto"

    if dt is None:
        dt = datetime.fromtimestamp(path.stat().st_mtime).date()
        print(f"  Note: using file mtime for {path.name} (add date to filename for accuracy)")

    return VideoRecord(patient_id=patient_id, date=dt, mode=mode, path=path)


def _parse_date(s: str):
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


def _date_from_text(s: str):
    for pat in (r"(\d{4})-(\d{2})-(\d{2})", r"(\d{4})(\d{2})(\d{2})"):
        m = re.search(pat, s)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
    return None


def _mode_from_text(s: str):
    s = s.lower()
    for mode in KNOWN_MODES - {"auto"}:
        if mode in s:
            return mode
    return None


def _label_sessions(records: list) -> list:
    records = sorted(records, key=lambda r: r.date)
    baseline = records[0].date
    for r in records:
        days = (r.date - baseline).days
        if days == 0:
            r.session_label = "baseline"
        else:
            r.session_label = f"wk{round(days / 7)}"
    return records


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old, 2)
        os.close(old)


def _video_duration(path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def run_analysis(record: VideoRecord, patient_dir: Path, resume: bool,
                 max_seconds: float = None) -> SessionResult:
    date_str    = record.date.strftime("%Y%m%d")
    session_dir = patient_dir / f"{date_str}_{record.session_label}_{record.path.stem}"
    mode        = record.mode.strip() or "auto"

    suffix_map = {
        "gait": "_gait.json", "tap": "_tap.json",
        "speech": "_speech.json", "flow": "_flow.json",
    }
    json_suffix = suffix_map.get(mode, "_tremor.json")
    json_path   = session_dir / (record.path.stem + json_suffix)

    if resume and json_path.exists():
        print(f"    [skip] already analysed")
        return SessionResult(record, session_dir, _extract(json_path), success=True)

    if max_seconds is not None:
        dur = _video_duration(record.path)
        if dur > max_seconds:
            print(f"    [skip] {dur:.0f}s > --max-seconds {max_seconds:.0f}s")
            return SessionResult(record, session_dir, {}, success=False,
                                 error=f"video too long ({dur:.0f}s)")

    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        if mode in ("hands", "feet", "face", "auto"):
            from tracker import track_video
            from analysis import analyze_tremor
            from report  import generate_report, save_outputs
            with _quiet():
                tracking = track_video(str(record.path), mode)
            result = analyze_tremor(tracking)
            generate_report(result, str(record.path))
            save_outputs(result, str(session_dir), str(record.path))

        elif mode == "gait":
            from gait import analyze_gait, generate_gait_report, save_gait_outputs
            with _quiet():
                result = analyze_gait(str(record.path))
            generate_gait_report(result, str(record.path))
            save_gait_outputs(result, str(session_dir), str(record.path))

        elif mode == "tap":
            from tap import analyze_taps, generate_tap_report, save_tap_outputs
            with _quiet():
                result = analyze_taps(str(record.path))
            generate_tap_report(result, str(record.path))
            save_tap_outputs(result, str(session_dir), str(record.path))

        elif mode == "flow":
            from flow import analyze_flow, generate_flow_report, save_flow_outputs, parse_roi
            roi_str = getattr(record, "roi", None)
            roi = parse_roi(roi_str) if roi_str else None
            result = analyze_flow(str(record.path), roi=roi)
            generate_flow_report(result, str(record.path))
            save_flow_outputs(result, str(session_dir), str(record.path))

        elif mode == "speech":
            from speech import (analyze_speech, generate_speech_report,
                                save_speech_outputs)
            with _quiet():
                result = analyze_speech(str(record.path))
            generate_speech_report(result, str(record.path))
            save_speech_outputs(result, str(session_dir), str(record.path))

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        metrics = _extract(json_path)
        return SessionResult(record, session_dir, metrics, success=True)

    except Exception as e:
        traceback.print_exc()
        return SessionResult(record, session_dir, {}, success=False, error=str(e))


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract(json_path: Path) -> dict:
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return {}

    mode = data.get("mode", "")
    m    = {}

    if mode in ("hands", "feet", "face"):
        p = data.get("primary", {})
        m["tremor_amplitude"]  = p.get("amplitude_normalized")
        m["tremor_freq_hz"]    = p.get("dominant_freq_hz")
        m["pd_band_power_pct"] = p.get("pd_band_power_pct")

    elif mode == "gait":
        segs = [s for s in data.get("segments", []) if s]
        if segs:
            def avg(key): return _nanmean([s.get(key) for s in segs])
            m["gait_cadence_spm"]   = avg("cadence_spm")
            m["gait_step_cv_pct"]   = avg("step_cv_pct")
            m["gait_arm_asymmetry"] = avg("arm_asymmetry")

    elif mode == "tap":
        m["tap_rate_hz"]        = data.get("mean_rate_hz")
        m["tap_rhythm_cv_pct"]  = data.get("rhythm_cv_pct")
        m["tap_arrests"]        = data.get("arrest_count")
        m["tap_rate_decrement"] = (data.get("rate_by_third")      or {}).get("decrement_pct")
        m["tap_amp_decrement"]  = (data.get("amplitude_by_third") or {}).get("decrement_pct")

    elif mode == "flow":
        m["tremor_amplitude"]  = data.get("amplitude")
        m["tremor_freq_hz"]    = data.get("dominant_freq_hz")
        m["pd_band_power_pct"] = data.get("pd_band_power_pct")

    elif mode == "speech":
        v = data.get("voice") or {}
        f = data.get("face")  or {}
        m["voice_loudness_db"]    = v.get("mean_loudness_db")
        m["voice_pitch_range_hz"] = v.get("pitch_range_hz")
        m["voice_tremor_pct"]     = v.get("vocal_tremor_power_pct")
        m["voice_long_pauses"]    = v.get("long_pause_count")
        m["face_mobility"]        = f.get("overall_mobility")
        m["face_asymmetry"]       = f.get("lr_asymmetry")

    return {k: v for k, v in m.items() if v is not None}


def _nanmean(vals):
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


# ---------------------------------------------------------------------------
# Longitudinal analysis
# ---------------------------------------------------------------------------

def longitudinal(sessions: list, patient_dir: Path, patient_id: str):
    # Check for mixed modes — metrics from different modes are not comparable
    modes = [s.record.mode for s in sessions]
    unique_modes = list(dict.fromkeys(modes))  # ordered, deduped
    mixed_modes = len(unique_modes) > 1
    if mixed_modes:
        print(f"  !! Mixed modes detected: {unique_modes}")
        print(f"     Amplitude and other absolute metrics are NOT comparable across modes.")
        print(f"     Longitudinal plot generated but treat cross-mode comparisons with caution.")

    all_keys = []
    for s in sessions:
        for k in s.metrics:
            if k not in all_keys:
                all_keys.append(k)

    if not all_keys:
        return

    labels   = [s.record.session_label for s in sessions]
    baseline = sessions[0].metrics
    series   = {k: [s.metrics.get(k) for s in sessions] for k in all_keys}

    # % change from baseline
    changes = {}
    for k, vals in series.items():
        bval = baseline.get(k)
        lval = vals[-1]
        if bval is not None and lval is not None and bval != 0:
            changes[k] = (lval - bval) / abs(bval) * 100
        else:
            changes[k] = None

    _plot_longitudinal(all_keys, series, labels, changes, patient_dir,
                       patient_id, mixed_modes, unique_modes)
    _save_longitudinal_json(sessions, changes, all_keys, patient_dir,
                            patient_id, mixed_modes, unique_modes)


def _plot_longitudinal(all_keys, series, labels, changes, patient_dir,
                       patient_id, mixed_modes=False, unique_modes=None):
    n  = len(all_keys)
    nc = 2
    nr = (n + 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(13, max(4, nr * 3.2)))
    title = f"Longitudinal Assessment — Patient {patient_id}"
    if mixed_modes:
        title += f"\n⚠ Mixed modes ({', '.join(unique_modes)}) — absolute metrics not comparable across modes"
    fig.suptitle(title, fontsize=12 if mixed_modes else 13, fontweight="bold",
                 color="crimson" if mixed_modes else "black")
    axes_flat = np.array(axes).flatten()

    for i, key in enumerate(all_keys):
        ax   = axes_flat[i]
        meta = METRIC_META.get(key, (key.replace("_", " ").title(), "", None))
        label, unit, better = meta

        vals    = series[key]
        x       = list(range(len(labels)))
        valid_x = [j for j, v in enumerate(vals) if v is not None]
        valid_y = [vals[j] for j in valid_x]

        if not valid_y:
            ax.axis("off")
            continue

        ax.plot(valid_x, valid_y, "o-", color="steelblue", lw=1.8, ms=7, zorder=3)
        ax.plot(valid_x[0], valid_y[0], "o", color="black", ms=10, zorder=5,
                label="Baseline")

        # Color final point
        chg = changes.get(key)
        pt_color = "steelblue"
        if chg is not None and better and len(valid_y) > 1:
            improved = (better == "lower"  and chg < -CHANGE_THRESHOLD_PCT) or \
                       (better == "higher" and chg >  CHANGE_THRESHOLD_PCT)
            worsened = (better == "lower"  and chg >  CHANGE_THRESHOLD_PCT) or \
                       (better == "higher" and chg < -CHANGE_THRESHOLD_PCT)
            pt_color = "#2ca02c" if improved else ("#d62728" if worsened else "steelblue")
            ax.plot(valid_x[-1], valid_y[-1], "o", color=pt_color, ms=11, zorder=6)

        if chg is not None and len(valid_y) > 1:
            arrow = "↑" if chg > 0 else "↓"
            ax.annotate(f"{arrow}{abs(chg):.1f}%",
                        xy=(valid_x[-1], valid_y[-1]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=9, color=pt_color, fontweight="bold")

        ax.set_xticks(valid_x)
        ax.set_xticklabels([labels[j] for j in valid_x], fontsize=8)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

    for i in range(len(all_keys), len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    out = patient_dir / "longitudinal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Longitudinal plot → {out.name}")


def _save_longitudinal_json(sessions, changes, all_keys, patient_dir,
                            patient_id, mixed_modes=False, unique_modes=None):
    baseline = sessions[0].metrics

    # Classify improvement
    improved, worsened, stable = [], [], []
    for k, chg in changes.items():
        if chg is None:
            continue
        better = METRIC_META.get(k, (None, None, None))[2]
        if better is None:
            continue
        if (better == "lower" and chg < -CHANGE_THRESHOLD_PCT) or \
           (better == "higher" and chg > CHANGE_THRESHOLD_PCT):
            improved.append(k)
        elif (better == "lower" and chg > CHANGE_THRESHOLD_PCT) or \
             (better == "higher" and chg < -CHANGE_THRESHOLD_PCT):
            worsened.append(k)
        else:
            stable.append(k)

    summary = {
        "patient_id":   patient_id,
        "n_sessions":   len(sessions),
        "mixed_modes":  mixed_modes,
        "modes_used":   unique_modes or [],
        "warning":      "Mixed measurement modes — absolute metric comparisons unreliable"
                        if mixed_modes else None,
        "sessions": [
            {
                "label":   s.record.session_label,
                "date":    s.record.date.isoformat(),
                "mode":    s.record.mode,
                "video":   s.record.path.name,
                "metrics": s.metrics,
                "success": s.success,
            }
            for s in sessions
        ],
        "baseline_metrics": baseline,
        "latest_metrics":   sessions[-1].metrics,
        "pct_change_from_baseline": {
            k: round(v, 2) if v is not None else None
            for k, v in changes.items()
        },
        "outcome": {
            "improved": improved,
            "worsened": worsened,
            "stable":   stable,
        },
    }

    out = patient_dir / "longitudinal_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"    Longitudinal summary → {out.name}")


# ---------------------------------------------------------------------------
# Batch summary CSV
# ---------------------------------------------------------------------------

def batch_summary(all_results: dict, output_dir: Path):
    rows = []
    for patient_id, sessions in sorted(all_results.items()):
        good = [s for s in sessions if s.success and s.metrics]
        if not good:
            continue
        baseline = good[0]
        latest   = good[-1]

        row = {
            "patient_id":    patient_id,
            "n_sessions":    len(good),
            "baseline_date": baseline.record.date.isoformat(),
            "latest_date":   latest.record.date.isoformat(),
            "days_observed": (latest.record.date - baseline.record.date).days,
        }

        all_keys = sorted({k for s in good for k in s.metrics})
        for key in all_keys:
            bval = baseline.metrics.get(key)
            lval = latest.metrics.get(key)
            row[f"baseline_{key}"] = round(bval, 4) if bval is not None else ""
            row[f"latest_{key}"]   = round(lval, 4) if lval is not None else ""
            if bval and lval and bval != 0:
                row[f"chg_pct_{key}"] = round((lval - bval) / abs(bval) * 100, 1)
            else:
                row[f"chg_pct_{key}"] = ""
        rows.append(row)

    if not rows:
        return

    all_cols = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in all_cols:
                all_cols.append(k)

    csv_path = output_dir / "batch_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n  Batch summary → {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Longitudinal batch processor for tremor/gait/speech analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video_dir",
                        help="Root directory to scan for videos (recursive)")
    parser.add_argument("--output-dir", default="batch_results",
                        help="Output root (default: ./batch_results)")
    parser.add_argument("--manifest",
                        help="CSV manifest: patient_id, date, mode, path")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sessions whose JSON output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be processed without running")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Skip videos longer than this many seconds (useful for "
                             "excluding full Zoom sessions from tap/gait analysis)")
    args = parser.parse_args()

    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    manifest   = Path(args.manifest) if args.manifest else None

    if not video_dir.exists():
        print(f"Error: not found: {video_dir}", file=sys.stderr)
        sys.exit(1)

    if manifest and not manifest.exists():
        print(f"Error: manifest not found: {manifest}", file=sys.stderr)
        print(f"  Generate one first with:", file=sys.stderr)
        print(f"    python make_manifest.py {video_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Discover ──────────────────────────────────────────────────────────────
    print(f"Scanning {video_dir} ...\n")
    records = scan(video_dir, manifest)

    if not records:
        print("No videos found.")
        sys.exit(0)

    patients = {}
    for r in records:
        patients.setdefault(r.patient_id, []).append(r)

    print(f"Found {len(records)} video(s) across {len(patients)} patient(s)\n")
    for pid, recs in sorted(patients.items()):
        recs = sorted(recs, key=lambda r: r.date)
        print(f"  {pid}  ({len(recs)} video(s))")
        for r in recs:
            print(f"    {r.date}  {r.mode:<8}  {r.path.name}")
    print()

    if args.dry_run:
        print("Dry run — no analysis run.")
        return

    # ── Process ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for patient_id, recs in sorted(patients.items()):
        patient_dir = output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)

        labeled = _label_sessions(recs)

        print(f"\n{'─'*56}")
        print(f"  Patient: {patient_id}  ({len(labeled)} session(s))")
        print(f"{'─'*56}")

        results = []
        for rec in labeled:
            print(f"\n  [{rec.session_label}]  {rec.path.name}  (mode: {rec.mode})")
            sr = run_analysis(rec, patient_dir, args.resume, args.max_seconds)
            results.append(sr)
            if sr.success and sr.metrics:
                for k, v in sr.metrics.items():
                    meta = METRIC_META.get(k, (k, "", None))
                    print(f"    {meta[0]:<26}  {v:>10.3f}  {meta[1]}")
            elif not sr.success:
                print(f"    FAILED: {sr.error[:80]}")

        good = [s for s in results if s.success and s.metrics]
        if len(good) > 1:
            print(f"\n  Longitudinal analysis for {patient_id} ...")
            longitudinal(good, patient_dir, patient_id)

        all_results[patient_id] = results

    batch_summary(all_results, output_dir)
    print("\nComplete.")


if __name__ == "__main__":
    main()
