#!/usr/bin/env python3
"""
Tremor Analyzer — frequency and amplitude from video for Parkinson's assessment.

Usage:
  python analyze.py patient_hand.mp4
  python analyze.py recording.mp4 --mode feet
  python analyze.py clip.mp4 --mode face --output-dir ./results
"""

import contextlib
import os
import sys

# Belt-and-suspenders: env vars first, then fd-level redirect during tracking
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path


@contextlib.contextmanager
def _quiet_stderr():
    """Redirect C++ stderr at the file-descriptor level to suppress MediaPipe noise."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old, 2)
        os.close(old)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tremor frequency and amplitude from a recorded video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--mode",
        choices=["auto", "hands", "feet", "face", "all", "gait"],
        default="auto",
        help="Body part to track, or 'all' to analyze every visible part (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for output files (default: <video_name>_tremor/ beside the video)",
    )
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"Error: file not found: {video}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTremor Analyzer")
    print(f"Video : {video.name}")

    try:
        from tracker import track_video, track_video_all
        from analysis import analyze_tremor
        from report import generate_report, save_outputs

        print("Tracking landmarks...")
        if args.mode == "gait":
            from gait import analyze_gait, generate_gait_report, save_gait_outputs
            with _quiet_stderr():
                analysis = analyze_gait(str(video))
            generate_gait_report(analysis, str(video))
            save_gait_outputs(analysis, args.output_dir, str(video))

        elif args.mode == "all":
            with _quiet_stderr():
                trackings = track_video_all(str(video))
            print("Analyzing tremor...")
            any_succeeded = False
            for tracking in trackings:
                try:
                    analysis = analyze_tremor(tracking)
                    generate_report(analysis, str(video))
                    save_outputs(analysis, args.output_dir, str(video), stem_prefix=tracking.mode)
                    any_succeeded = True
                except ValueError as e:
                    print(f"  Skipping {tracking.mode}: {e}")
            if not any_succeeded:
                raise ValueError("No body parts yielded usable tremor data.")
        else:
            with _quiet_stderr():
                tracking = track_video(str(video), args.mode)
            print("Analyzing tremor...")
            analysis = analyze_tremor(tracking)
            generate_report(analysis, str(video))
            save_outputs(analysis, args.output_dir, str(video))

    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
