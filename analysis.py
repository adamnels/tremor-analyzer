import numpy as np
from scipy import signal, interpolate
from dataclasses import dataclass
from typing import Optional

PD_LOW = 3.0   # Hz — lower bound of PD resting tremor band
PD_HIGH = 6.0  # Hz — upper bound
ANALYSIS_BAND = (2.0, 15.0)  # Hz — anything below 2 Hz is drift/voluntary movement
MIN_VALID_SECONDS = 2.0      # minimum tracked frames required


@dataclass
class LandmarkAnalysis:
    name: str
    time: np.ndarray          # seconds, shape (N,)
    displacement: np.ndarray  # primary-axis filtered signal, normalized, shape (N,)
    dominant_freq: float      # Hz
    amplitude: float          # peak-to-peak estimate, normalized to reference size
    tremor_index: float       # fraction of tremor power in PD band
    in_pd_range: bool
    freqs: np.ndarray         # FFT frequency bins
    power: np.ndarray         # combined x+y power spectrum
    spectrogram_t: np.ndarray
    spectrogram_f: np.ndarray
    spectrogram_p: np.ndarray


@dataclass
class TremorAnalysis:
    mode: str
    fps: float
    duration: float
    detection_rate: float
    landmarks: list           # list[LandmarkAnalysis]
    primary: LandmarkAnalysis  # landmark with highest amplitude
    deviation_flags: list     # list[str]


def analyze_tremor(tracking) -> TremorAnalysis:
    ref_size = float(np.nanmedian(tracking.reference_size))
    if np.isnan(ref_size) or ref_size < 1.0:
        ref_size = 100.0
        print("  Warning: reference size unavailable — amplitude reported in pixels")

    results = []
    for name, positions in tracking.landmarks.items():
        r = _analyze_landmark(name, positions, tracking.fps, ref_size)
        if r is not None:
            results.append(r)

    if not results:
        raise ValueError(
            "Insufficient tracking data — video may be too short or body part not visible."
        )

    primary = max(results, key=lambda r: r.amplitude)
    flags = _generate_flags(results, primary, tracking.fps, tracking.detection_rate)

    return TremorAnalysis(
        mode=tracking.mode,
        fps=tracking.fps,
        duration=tracking.frame_count / tracking.fps,
        detection_rate=tracking.detection_rate,
        landmarks=results,
        primary=primary,
        deviation_flags=flags,
    )


def _analyze_landmark(
    name: str, positions: np.ndarray, fps: float, ref_size: float
) -> Optional[LandmarkAnalysis]:
    x, y = positions[:, 0], positions[:, 1]
    valid = ~(np.isnan(x) | np.isnan(y))

    if valid.sum() < fps * MIN_VALID_SECONDS:
        return None

    t = np.arange(len(x)) / fps

    if valid.sum() < len(x):
        x = _fill_nans(t, x, valid)
        y = _fill_nans(t, y, valid)

    # Remove DC offset and normalize to reference size
    dx = (x - x.mean()) / ref_size
    dy = (y - y.mean()) / ref_size

    # Bandpass filter both axes
    nyq = fps / 2.0
    low = ANALYSIS_BAND[0] / nyq
    high = min(ANALYSIS_BAND[1], nyq * 0.95) / nyq
    if low >= high:
        return None

    b, a = signal.butter(4, [low, high], btype="band")
    try:
        dx_f = signal.filtfilt(b, a, dx)
        dy_f = signal.filtfilt(b, a, dy)
    except ValueError:
        return None  # signal too short for filter pad length

    # Primary axis: whichever has more variance after filtering
    primary_disp = dx_f if dx_f.std() >= dy_f.std() else dy_f

    # Combined power spectrum (captures motion in any direction)
    freqs = np.fft.rfftfreq(len(dx_f), 1.0 / fps)
    power = np.abs(np.fft.rfft(dx_f)) ** 2 + np.abs(np.fft.rfft(dy_f)) ** 2

    band_mask = (freqs >= ANALYSIS_BAND[0]) & (freqs <= ANALYSIS_BAND[1])
    if not band_mask.any():
        return None

    dominant_freq = float(freqs[band_mask][np.argmax(power[band_mask])])

    pd_mask = (freqs >= PD_LOW) & (freqs <= PD_HIGH)
    pd_power = float(power[pd_mask].sum())
    total_power = float(power[band_mask].sum())
    tremor_index = pd_power / (total_power + 1e-12)

    # Peak-to-peak amplitude estimate from 2D RMS
    amplitude = float(np.sqrt(dx_f.var() + dy_f.var()) * 2 * np.sqrt(2))

    # Spectrogram for temporal frequency tracking
    nperseg = max(16, min(int(fps * 2), len(primary_disp) // 2))
    noverlap = int(nperseg * 0.75)
    f_spec, t_spec, Sxx = signal.spectrogram(
        primary_disp, fs=fps, nperseg=nperseg, noverlap=noverlap, window="hann"
    )

    return LandmarkAnalysis(
        name=name,
        time=t,
        displacement=primary_disp,
        dominant_freq=dominant_freq,
        amplitude=amplitude,
        tremor_index=tremor_index,
        in_pd_range=PD_LOW <= dominant_freq <= PD_HIGH,
        freqs=freqs,
        power=power,
        spectrogram_t=t_spec,
        spectrogram_f=f_spec,
        spectrogram_p=Sxx,
    )


def _fill_nans(t: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    f = interpolate.interp1d(
        t[valid], v[valid], kind="linear",
        bounds_error=False, fill_value=(v[valid][0], v[valid][-1]),
    )
    return f(t)


def _generate_flags(results, primary: LandmarkAnalysis, fps: float, detection_rate: float = 1.0) -> list:
    flags = []

    if detection_rate < 0.5:
        flags.append(
            f"LOW DETECTION RATE: body part visible in only {detection_rate*100:.0f}% of frames "
            "— results may be unreliable. Try better lighting or a tighter frame."
        )

    if not primary.in_pd_range:
        flags.append(
            f"FREQUENCY OUT OF PD RANGE: dominant {primary.dominant_freq:.2f} Hz "
            f"(PD resting tremor expected {PD_LOW}–{PD_HIGH} Hz)"
        )

    if primary.tremor_index < 0.3:
        flags.append(
            f"LOW PD-BAND POWER: only {primary.tremor_index*100:.0f}% of tremor energy "
            f"falls in {PD_LOW}–{PD_HIGH} Hz — pattern may be atypical or motion is noise-dominated"
        )

    # Frequency stability across spectrogram windows
    f = primary.spectrogram_f
    s = primary.spectrogram_p
    band_mask = (f >= ANALYSIS_BAND[0]) & (f <= ANALYSIS_BAND[1])
    if band_mask.any() and s.shape[1] >= 4:
        dom_per_win = f[band_mask][np.argmax(s[band_mask], axis=0)]
        freq_std = float(dom_per_win.std())
        freq_mean = float(dom_per_win.mean())
        if freq_std > 0.8:
            flags.append(
                f"FREQUENCY INSTABILITY: {freq_mean:.1f} ± {freq_std:.1f} Hz across "
                "recording windows — tremor pattern is irregular"
            )

    return flags
