"""Trim raw accident videos down to the exact 2-5 second accident window.

Reads an annotations CSV (filename,start_sec,end_sec) and cuts each raw video
in a source directory to just the annotated window using ffmpeg, so the
training set contains only the accident/impact itself rather than the normal
driving footage before and after it.

CSV format (header required):
    filename,start_sec,end_sec
    crash_001.mp4,12.0,15.5

Usage:
    python trim_accident_clips.py \
        --annotations dataset/train/annotations.csv \
        --source-dir dataset/train/accident_raw \
        --output-dir dataset/train/accident \
        --dry-run

Drop --dry-run to actually invoke ffmpeg and write trimmed clips.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys

MIN_DURATION = 2.0
MAX_DURATION = 5.0


def read_annotations(annotations_path):
    rows = []
    with open(annotations_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"filename", "start_sec", "end_sec"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"annotations CSV must have columns {sorted(required)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            rows.append(
                {
                    "filename": row["filename"].strip(),
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                }
            )
    return rows


def validate_row(row, source_dir):
    problems = []
    src_path = os.path.join(source_dir, row["filename"])
    if not os.path.isfile(src_path):
        problems.append(f"source file not found: {src_path}")

    duration = row["end_sec"] - row["start_sec"]
    if duration <= 0:
        problems.append(f"end_sec ({row['end_sec']}) must be after start_sec ({row['start_sec']})")
    elif duration < MIN_DURATION or duration > MAX_DURATION:
        problems.append(
            f"duration {duration:.2f}s is outside the required "
            f"{MIN_DURATION}-{MAX_DURATION}s accident window"
        )
    return problems


def trim_clip(src_path, out_path, start_sec, end_sec):
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", src_path,
        "-t", str(duration),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="CSV with filename,start_sec,end_sec")
    parser.add_argument("--source-dir", required=True, help="Directory containing the raw source videos")
    parser.add_argument("--output-dir", required=True, help="Directory to write trimmed clips into")
    parser.add_argument("--dry-run", action="store_true", help="Validate annotations without running ffmpeg")
    args = parser.parse_args()

    if not args.dry_run and shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH. Install it (e.g. apt-get install ffmpeg) or run with --dry-run.", file=sys.stderr)
        sys.exit(1)

    rows = read_annotations(args.annotations)
    if not rows:
        print("No rows found in annotations CSV.")
        return

    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    ok_count = 0
    fail_count = 0
    for row in rows:
        problems = validate_row(row, args.source_dir)
        if problems:
            fail_count += 1
            print(f"[INVALID] {row['filename']}: " + "; ".join(problems))
            continue

        ok_count += 1
        if args.dry_run:
            duration = row["end_sec"] - row["start_sec"]
            print(f"[OK] {row['filename']} -> {duration:.2f}s window [{row['start_sec']}, {row['end_sec']}]")
            continue

        src_path = os.path.join(args.source_dir, row["filename"])
        out_path = os.path.join(args.output_dir, row["filename"])
        try:
            trim_clip(src_path, out_path, row["start_sec"], row["end_sec"])
            print(f"[TRIMMED] {row['filename']} -> {out_path}")
        except subprocess.CalledProcessError as e:
            fail_count += 1
            ok_count -= 1
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            print(f"[FFMPEG ERROR] {row['filename']}: {stderr.strip()[-500:]}")

    print(f"\n{ok_count} valid, {fail_count} invalid, out of {len(rows)} total rows.")
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
