import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# Landmark indices and reference pairs (for spatial normalization)
HAND_LANDMARKS = {
    "wrist": 0,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
HAND_REF = (0, 9)   # wrist → middle MCP (~4 cm on an adult hand)

FOOT_LANDMARKS = {
    "left_ankle": 27,
    "right_ankle": 28,
}
FOOT_REF = (23, 24)  # left hip → right hip

FACE_LANDMARKS = {
    "nose_tip": 4,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
}
FACE_REF = (33, 263)  # outer eye-to-eye distance


@dataclass
class TrackingResult:
    mode: str
    fps: float
    frame_count: int
    landmarks: dict          # name → (N, 2) pixel coords (NaN when not detected)
    reference_size: np.ndarray  # (N,) reference distance per frame
    detection_rate: float    # fraction of frames where body part was found


def track_video(video_path: str, mode: str = "auto") -> TrackingResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps < 15:
        print(f"  Warning: low frame rate ({fps:.1f} fps) — may miss fast tremor components")
    if total / fps < 3:
        print(f"  Warning: short video ({total/fps:.1f}s) — recommend ≥5 s for reliable analysis")

    if mode == "auto":
        mode = _detect_mode(video_path, total)

    print(f"  Mode: {mode}  |  {fps:.1f} fps  |  ~{total/fps:.0f}s  ({total} frames)")

    if mode == "hands":
        return _track_hands(video_path, fps)
    elif mode == "feet":
        return _track_feet(video_path, fps)
    elif mode == "face":
        return _track_face(video_path, fps)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _detect_mode(video_path: str, total_frames: int) -> str:
    cap = cv2.VideoCapture(video_path)
    indices = set(np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    scores = {"hands": 0, "feet": 0, "face": 0}

    with mp_hands.Hands(static_image_mode=True) as h:
        for f in frames:
            if h.process(f).multi_hand_landmarks:
                scores["hands"] += 1

    with mp_pose.Pose(static_image_mode=True) as p:
        for f in frames:
            r = p.process(f)
            if r.pose_landmarks:
                lms = r.pose_landmarks.landmark
                if lms[27].visibility > 0.5 or lms[28].visibility > 0.5:
                    scores["feet"] += 1

    with mp_face.FaceMesh(static_image_mode=True) as fm:
        for f in frames:
            if fm.process(f).multi_face_landmarks:
                scores["face"] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        raise ValueError(
            "No body part detected. Specify --mode (hands / feet / face) manually."
        )

    print(f"  Auto-detected: {best} ({scores[best]}/{len(frames)} sample frames)")
    return best


def _track_hands(video_path: str, fps: float) -> TrackingResult:
    data = {k: [] for k in HAND_LANDMARKS}
    ref_sizes = []
    n = 0
    detected = 0

    cap = cv2.VideoCapture(video_path)
    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            h, w = frame.shape[:2]
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_hand_landmarks:
                detected += 1
                lms = result.multi_hand_landmarks[0].landmark
                for name, idx in HAND_LANDMARKS.items():
                    data[name].append([lms[idx].x * w, lms[idx].y * h])
                a = np.array([lms[HAND_REF[0]].x * w, lms[HAND_REF[0]].y * h])
                b = np.array([lms[HAND_REF[1]].x * w, lms[HAND_REF[1]].y * h])
                ref_sizes.append(np.linalg.norm(b - a))
            else:
                for name in HAND_LANDMARKS:
                    data[name].append([np.nan, np.nan])
                ref_sizes.append(np.nan)

            if n % max(1, int(fps)) == 0:
                print(f"\r  Tracking... {n/fps:.0f}s", end="", flush=True)

    cap.release()
    rate = detected / n if n > 0 else 0.0
    print(f"\r  Tracked {n/fps:.0f}s — hands detected in {rate*100:.0f}% of frames")
    return TrackingResult(
        mode="hands", fps=fps, frame_count=n,
        landmarks={k: np.array(v) for k, v in data.items()},
        reference_size=np.array(ref_sizes),
        detection_rate=rate,
    )


def _track_feet(video_path: str, fps: float) -> TrackingResult:
    data = {k: [] for k in FOOT_LANDMARKS}
    ref_sizes = []
    n = 0
    detected = 0

    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
        static_image_mode=False,
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
                for name, idx in FOOT_LANDMARKS.items():
                    vis = lms[idx].visibility
                    if vis > 0.3:
                        data[name].append([lms[idx].x * w, lms[idx].y * h])
                    else:
                        data[name].append([np.nan, np.nan])
                a = np.array([lms[FOOT_REF[0]].x * w, lms[FOOT_REF[0]].y * h])
                b = np.array([lms[FOOT_REF[1]].x * w, lms[FOOT_REF[1]].y * h])
                ref_sizes.append(np.linalg.norm(b - a))
            else:
                for name in FOOT_LANDMARKS:
                    data[name].append([np.nan, np.nan])
                ref_sizes.append(np.nan)

            if n % max(1, int(fps)) == 0:
                print(f"\r  Tracking... {n/fps:.0f}s", end="", flush=True)

    cap.release()
    rate = detected / n if n > 0 else 0.0
    print(f"\r  Tracked {n/fps:.0f}s — pose detected in {rate*100:.0f}% of frames")
    return TrackingResult(
        mode="feet", fps=fps, frame_count=n,
        landmarks={k: np.array(v) for k, v in data.items()},
        reference_size=np.array(ref_sizes),
        detection_rate=rate,
    )


def _track_face(video_path: str, fps: float) -> TrackingResult:
    data = {k: [] for k in FACE_LANDMARKS}
    ref_sizes = []
    n = 0
    detected = 0

    cap = cv2.VideoCapture(video_path)
    with mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as face:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            h, w = frame.shape[:2]
            result = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_face_landmarks:
                detected += 1
                lms = result.multi_face_landmarks[0].landmark
                for name, idx in FACE_LANDMARKS.items():
                    data[name].append([lms[idx].x * w, lms[idx].y * h])
                a = np.array([lms[FACE_REF[0]].x * w, lms[FACE_REF[0]].y * h])
                b = np.array([lms[FACE_REF[1]].x * w, lms[FACE_REF[1]].y * h])
                ref_sizes.append(np.linalg.norm(b - a))
            else:
                for name in FACE_LANDMARKS:
                    data[name].append([np.nan, np.nan])
                ref_sizes.append(np.nan)

            if n % max(1, int(fps)) == 0:
                print(f"\r  Tracking... {n/fps:.0f}s", end="", flush=True)

    cap.release()
    rate = detected / n if n > 0 else 0.0
    print(f"\r  Tracked {n/fps:.0f}s — face detected in {rate*100:.0f}% of frames")
    return TrackingResult(
        mode="face", fps=fps, frame_count=n,
        landmarks={k: np.array(v) for k, v in data.items()},
        reference_size=np.array(ref_sizes),
        detection_rate=rate,
    )
