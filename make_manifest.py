#!/usr/bin/env python3
"""
Generate a draft manifest CSV from a nested patient video directory.

Usage:
  python make_manifest.py PATIENT_DIR [--output manifest.csv]

Opens the CSV in Excel / Numbers to fill in the 'mode' column, then run:
  python batch.py PATIENT_DIR --manifest manifest.csv --output-dir results
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path


VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".MP4", ".MOV"}

KNOWN_MODES = {"hands", "feet", "face", "gait", "tap", "speech"}

# Top-level folders that are not patients
SKIP_FOLDERS = {"Patient Summary", "untitled folder", "Redacted Emails", ".DS_Store"}

# Filename patterns that suggest a Zoom recording
ZOOM_PATTERNS = re.compile(r"zoom|gmt|meet|intake|intro|session|visit", re.IGNORECASE)


def scan(root: Path) -> list:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in VIDEO_EXT:
            continue

        rel   = path.relative_to(root)
        parts = rel.parts

        # Skip non-patient top-level folders
        if parts[0] in SKIP_FOLDERS:
            continue

        patient_id = parts[0]
        date_str, date_source = _get_date(path, root)
        mode       = _get_mode(path, root)
        notes      = _get_notes(path, date_source)

        rows.append({
            "patient_id": patient_id,
            "date":       date_str,
            "mode":       mode,
            "path":       str(path.resolve()),
            "notes":      notes,
        })

    return rows


def _get_date(path: Path, root: Path) -> tuple:
    """Return (date_string, source) where source describes how it was found."""
    rel_str = str(path.relative_to(root))

    # Try ISO date with optional time: 2022-08-20 or 2022-08-20 16-32-05
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", rel_str)
    if m:
        try:
            datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "filename"
        except ValueError:
            pass

    # Try compact date: WIN_20240309_... or standalone 20240309
    m = re.search(r"(\d{4})(\d{2})(\d{2})", rel_str)
    if m:
        try:
            datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "filename"
        except ValueError:
            pass

    # Fall back to file modification time
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime.strftime("%Y-%m-%d"), "mtime"


def _get_mode(path: Path, root: Path) -> str:
    """Return mode if detectable from path, else blank for user to fill in."""
    text = str(path.relative_to(root)).lower()
    for mode in KNOWN_MODES:
        if mode in text:
            return mode
    return ""


def _get_notes(path: Path, date_source: str) -> str:
    notes = []

    if date_source == "mtime":
        notes.append("NO DATE IN FILENAME — date is file mtime, verify before running")

    stem = path.stem.lower()
    if ZOOM_PATTERNS.search(stem) or path.suffix.lower() == ".mov" and len(stem) < 12:
        notes.append("possible Zoom/multi-person — check video before running")

    if re.match(r"img_\d+", stem, re.IGNORECASE):
        notes.append("iPhone camera roll — no reliable date")

    if not _get_mode(path, path.parent.parent):
        notes.append("mode not detected — fill in mode column")

    return "  |  ".join(notes)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a draft manifest CSV for batch processing.",
    )
    parser.add_argument("patient_dir", help="Root patient directory to scan")
    parser.add_argument("--output", default="manifest.csv",
                        help="Output CSV path (default: manifest.csv)")
    args = parser.parse_args()

    root = Path(args.patient_dir)
    if not root.exists():
        print(f"Error: not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {root} ...")
    rows = scan(root)

    if not rows:
        print("No video files found.")
        sys.exit(0)

    out = Path(args.output)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "date", "mode", "path", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    patients = sorted({r["patient_id"] for r in rows})
    no_mode  = [r for r in rows if not r["mode"]]
    no_date  = [r for r in rows if "mtime" in r["notes"]]
    zoom     = [r for r in rows if "Zoom" in r["notes"] or "multi-person" in r["notes"]]

    print(f"\nManifest written → {out.resolve()}")
    print(f"\n  {len(rows)} video(s) across {len(patients)} patient(s)")
    print(f"  {len(no_mode)} need mode filled in")
    print(f"  {len(no_date)} have no date in filename (mtime used — verify)")
    print(f"  {len(zoom)} flagged as possible multi-person / Zoom")
    print(f"\nPatients found:")
    for p in patients:
        count = sum(1 for r in rows if r["patient_id"] == p)
        print(f"  {p:<30} {count} video(s)")

    print(f"\nNext steps:")
    print(f"  1. Open {out.name} in Excel or Numbers")
    print(f"  2. Fill in the 'mode' column for each row:")
    print(f"     hands  feet  face  gait  tap  speech  auto")
    print(f"  3. Delete rows for videos you don't want to process")
    print(f"  4. Run: python batch.py {root} --manifest {out} --output-dir results --dry-run")
    print(f"  5. If the dry-run looks right, remove --dry-run")


if __name__ == "__main__":
    main()
