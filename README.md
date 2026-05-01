# Tremor & Gait Analyzer

A Python CLI tool for measuring Parkinson's disease motor symptoms from recorded video. Quantifies tremor frequency and amplitude (hands, feet, face) and gait characteristics (cadence, arm swing, step regularity) without any wearable sensors.

---

## Installation

```bash
pip install mediapipe opencv-python numpy scipy matplotlib
```

Requires Python 3.10+. Tested on macOS (Apple Silicon). Processing runs at approximately 1–3× real-time on modern hardware.

---

## Usage

```
python analyze.py VIDEO [--mode MODE] [--output-dir DIR]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `VIDEO` | required | Path to the video file |
| `--mode` | `auto` | See mode table below |
| `--output-dir` | next to video | Directory for output files |

### Modes

| Mode | Description |
|---|---|
| `auto` | Sample 10 frames and pick whichever body part has the most detections |
| `hands` | Track wrist and fingertips for hand tremor |
| `feet` | Track ankles via full-body pose for foot tremor |
| `face` | Track nose, chin, and eye landmarks for facial/jaw tremor |
| `all` | Single pass running all three tremor models simultaneously; analyzes every body part detected in ≥20% of frames |
| `gait` | Full-body pose tracking for walking analysis; auto-detects turnaround |

### Examples

```bash
# Tremor — auto-detect body part
python analyze.py patient_hand.mp4

# Tremor — specify mode
python analyze.py recording.mp4 --mode hands
python analyze.py recording.mp4 --mode face

# Tremor — analyze all visible body parts in one pass
python analyze.py recording.mp4 --mode all

# Gait — patient walks toward/away from camera
python analyze.py walk.mp4 --mode gait

# Save outputs to a specific directory
python analyze.py recording.mp4 --mode gait --output-dir ./results/patient_01
```

### Output files

For tremor modes, outputs go to `<video_name>_tremor/`:

| File | Contents |
|---|---|
| `<stem>_tremor.png` | 6-panel figure (see below) |
| `<stem>_tremor.json` | Machine-readable summary |
| `<stem>_landmarks.png` | Per-landmark comparison (if multiple landmarks tracked) |

For `--mode all`, each detected body part gets its own files:
`hands_<stem>_tremor.png`, `face_<stem>_tremor.png`, etc.

For `--mode gait`, outputs go to `<video_name>_gait/`:

| File | Contents |
|---|---|
| `<stem>_gait.png` | 5-panel figure (see below) |
| `<stem>_gait.json` | Machine-readable summary |

---

## Clinical Reference

### Parkinson's Resting Tremor

PD resting tremor is a 3–6 Hz oscillation present when the limb is at rest and suppressed by voluntary movement. It is the most common presenting symptom in PD and is typically distal and asymmetric in early disease.

**What the tool measures:**
- Dominant tremor frequency (Hz)
- Amplitude, normalized to an anatomical reference distance (see below)
- Fraction of total tremor power that falls within the 3–6 Hz PD band
- How the dominant frequency varies across the recording (temporal stability)

**Flags raised:**

| Flag | Meaning |
|---|---|
| `FREQUENCY OUT OF PD RANGE` | Dominant frequency outside 3–6 Hz. Could indicate essential tremor (4–12 Hz), physiological tremor (8–12 Hz), or a non-tremor movement artifact. |
| `LOW PD-BAND POWER` | Less than 30% of tremor energy is in the 3–6 Hz band. The motion being measured may not be a true tremor, or may be dominated by voluntary movement or noise. |
| `FREQUENCY INSTABILITY` | Standard deviation of the dominant frequency across 2-second spectrogram windows exceeds 0.8 Hz. May reflect irregular tremor, medication fluctuation (on/off states), or poor landmark tracking. |
| `LOW DETECTION RATE` | The body part was visible in fewer than 50% of frames. Results should be treated with caution. |

**Amplitude reference distances** (used to normalize pixel measurements):

| Mode | Reference | Approximate physical size |
|---|---|---|
| `hands` | Wrist to middle finger MCP joint | ~4 cm (adult) |
| `feet` | Hip width (left to right hip landmark) | ~30 cm (adult) |
| `face` | Outer eye-to-outer eye distance | ~6 cm (adult) |

Amplitude is expressed as a dimensionless ratio (e.g., `0.15 × wrist-to-MCP`). To convert to approximate centimeters, multiply by the physical reference size.

**Notes by body part:**

- **Hands** — Most sensitive site for PD resting tremor. The tool tracks wrist, index, middle, ring, and pinky tips. Fingertips (especially index and middle) typically show the cleanest tremor signal. The wrist often shows lower frequency due to anchoring; the pinky is the least reliably tracked.
- **Feet** — Foot tremor occurs in PD but is less common than hand tremor. Requires both ankles to be visible with good pose estimation. May not be detectable if the patient is seated with legs supported.
- **Face** — Jaw tremor (4–6 Hz) is a recognized PD feature. Head tremor at 3–6 Hz can occur in PD but is more characteristic of essential tremor. Face tremor amplitudes are much smaller than limb tremors; detection requires good lighting and a relatively close frame.

---

### Gait (`--mode gait`)

PD gait is characterized by reduced stride length, reduced arm swing (often asymmetric), increased cadence, and step-to-step irregularity. Festination — involuntary acceleration with progressively shorter steps — is a more advanced feature.

The tool expects a video of the patient walking toward and/or away from the camera. A single direction also works.

**What the tool measures:**

| Metric | Description |
|---|---|
| **Cadence** | Steps per minute, derived from the frequency of hip vertical oscillation |
| **Step CV** | Coefficient of variation of step intervals (%) — a measure of gait rhythm regularity |
| **Arm swing (L/R)** | RMS amplitude of lateral wrist oscillation, normalized to shoulder width |
| **Arm asymmetry** | Ratio of larger to smaller arm swing (max/min). A ratio of 1.0 means perfectly symmetric. |

**Flags raised:**

| Flag | Threshold | Clinical context |
|---|---|---|
| `ARM SWING ASYMMETRY` | Ratio > 1.5× | Asymmetric arm swing is one of the earliest and most specific signs of PD, often preceding other motor symptoms. It reflects the asymmetric dopaminergic deficit typical of early PD. |
| `REDUCED ARM SWING` | Normalized amplitude < 0.06 | Bilaterally reduced arm swing (hypomimia of gait) is characteristic of more advanced or bilateral PD. Also seen in depression and certain medications. |
| `IRREGULAR GAIT` | Step CV > 4% | Increased step-to-step variability is associated with fall risk in PD and correlates with disease severity. Normal healthy adults have step CV < 2–3%. |
| `HIGH CADENCE` | > 130 steps/min | Elevated cadence combined with visually short steps may indicate festination. Normal comfortable walking is typically 100–120 steps/min. |

**Turnaround detection:** The tool uses the apparent size of the pose skeleton (shoulder-to-ankle pixel distance) to detect when the patient reverses direction. When found, the video is split into two segments (e.g., "toward" and "away") analyzed independently. Each segment gets its own cadence and arm swing measurements in the report.

**Limitation — forward lean:** Stooped posture (camptocormia) is a recognized PD feature but cannot be reliably measured from a front- or back-facing camera. A lateral (side-view) recording would be required.

---

## Methodology

### Landmark Tracking

All tracking uses [MediaPipe](https://developers.google.com/mediapipe):

| Mode | Model | Landmarks used |
|---|---|---|
| `hands` | MediaPipe Hands | Wrist (0), fingertips (4, 8, 12, 16, 20) |
| `feet` | MediaPipe Pose | Ankles (27, 28); visibility threshold 0.3 |
| `face` | MediaPipe Face Mesh | Nose tip (4), chin (152), outer eye corners (33, 263) |
| `gait` | MediaPipe Pose (complexity 1) | Shoulders (11, 12), hips (23, 24), knees (25, 26), ankles (27, 28), wrists (15, 16) |

Frames where the body part is not detected are recorded as NaN and filled by linear interpolation before analysis. Detection rate (% of frames with a valid detection) is reported and flagged if below 50%.

### Tremor Signal Processing

1. **DC removal** — subtract the mean position to remove the resting location of each landmark.

2. **Normalization** — divide by the anatomical reference distance (computed per-frame, median taken over the clip) to produce a dimensionless displacement signal.

3. **Bandpass filtering** — 4th-order Butterworth filter, 2–15 Hz. The 2 Hz lower cutoff removes slow voluntary movements and interpolation artifacts from tracking gaps; the 15 Hz upper cutoff removes high-frequency noise. Applied with `scipy.signal.filtfilt` (zero-phase).

4. **Primary axis selection** — x and y displacements are filtered independently. The axis with greater post-filter variance is used for the time-series display. The combined power spectrum (|FFT(dx)|² + |FFT(dy)|²) is used for frequency analysis, which captures tremor in any direction without requiring axis alignment.

5. **Dominant frequency** — the frequency bin with maximum combined power in the 2–15 Hz band.

6. **PD-band power** — fraction of total tremor power (2–15 Hz) that falls within 3–6 Hz.

7. **Amplitude** — 2D RMS of the filtered signal scaled to peak-to-peak: `√(var(dx_f) + var(dy_f)) × 2√2`. Normalized to the anatomical reference distance.

8. **Spectrogram** — short-time FFT using 2-second Hann windows with 75% overlap (`scipy.signal.spectrogram`). Used to track how the dominant frequency changes over the recording.

9. **Frequency instability** — for each spectrogram window, find the frequency bin with maximum power in 2–15 Hz. Report mean ± std across all windows.

### Gait Signal Processing

1. **Detrending** — the hip y-position and wrist x-positions are linearly detrended within each walking segment. This removes the systematic change in apparent position as the patient approaches or recedes from the camera.

2. **Cadence** — 4th-order Butterworth bandpass filter (0.5–3.0 Hz = 30–180 steps/min) applied to detrended hip y-position. Dominant frequency from FFT, converted to steps/min (Hz × 60). The hip oscillates vertically once per step.

3. **Step detection** — `scipy.signal.find_peaks` on the filtered hip signal with minimum inter-peak distance corresponding to the upper cadence limit. Inter-peak intervals give step regularity (CV).

4. **Arm swing** — lateral (x-axis) wrist position, detrended and bandpass filtered at the same cadence band. RMS amplitude × 2√2 gives a peak-to-peak estimate. Normalized by the median shoulder width (inter-shoulder pixel distance) to cancel out the effect of camera distance — as the patient gets closer, both arm swing pixels and shoulder width pixels scale proportionally.

5. **Turnaround detection** — the shoulder-to-ankle pixel distance is a proxy for how large the person appears in frame. Smoothed with a 1.5-second moving average, its derivative sign is checked in the first and last thirds of the video. A sign reversal in the middle 60% of the video marks the turnaround. Direction labels ("toward"/"away") are assigned by whether the skeleton was growing or shrinking before the turnaround.

---

## Technical Notes

- **Minimum video length** — 5 seconds recommended for tremor; at least 8–10 seconds for gait (to allow turnaround detection and at least 3 seconds per walking segment).
- **Frame rate** — 30 fps is ideal. Below 15 fps a warning is issued; the Nyquist frequency at 15 fps is 7.5 Hz, which still covers the PD tremor band but limits the upper range.
- **Camera distance** — tremor amplitude and gait arm swing are both normalized to anatomical reference distances, making them approximately comparable across sessions filmed at different distances.
- **Lighting** — MediaPipe landmark detection degrades significantly in low light. Ensure the patient is well lit against a reasonably uncluttered background.
- **`--mode all`** — runs all three tremor models simultaneously in a single video pass for efficiency. Each is approximately as accurate as its single-mode counterpart.
- **Logging** — MediaPipe's C++ logging is suppressed at the file-descriptor level so it does not appear in the terminal output.
- **Patient data** — video files and output directories are listed in `.gitignore` and will not be committed to version control.

---

## Limitations

- **2D video only** — no depth information. Amplitude estimates assume the tremor is primarily in the plane of the camera. Tremor directed toward/away from the camera will be underestimated.
- **No absolute stride length** — gait stride length cannot be computed in physical units (meters) without knowing the camera-to-subject distance or having a calibration reference in frame.
- **No forward lean measurement** — stooped posture requires a side-view camera.
- **Not a diagnostic instrument** — outputs are quantitative features intended to support clinical assessment and longitudinal monitoring, not to produce or replace a diagnosis.
