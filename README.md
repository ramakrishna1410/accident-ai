# accident-ai

Video-based accident detection: a `torchvision` `r3d_18` (3D-CNN) classifier
fine-tuned to distinguish "accident" vs "normal" video clips, plus a YOLOv8
based heuristic detector (`src/test.py`, `src/testvideo.py`, `src/testvideoV2.py`).

## Dataset layout

```
dataset/
  train/
    accident/   # trimmed 2-5s accident-impact clips (*.mp4)
    normal/     # normal driving clips (*.mp4)
  val/
    accident/
    normal/
```

`dataset/` is gitignored — populate it locally.

## Preparing accident clips (trim to the exact 2-5s window)

Raw source footage usually contains normal driving before/after the
accident. Before training, trim each raw accident video down to just the
impact window:

1. Watch each raw video once and note the accident start/end seconds.
2. Fill in an annotations CSV (see `src/annotations.csv.example`):
   ```
   filename,start_sec,end_sec
   crash_001.mp4,12.0,15.5
   ```
3. Validate the annotations without touching any files:
   ```
   python src/trim_accident_clips.py \
     --annotations dataset/train/annotations.csv \
     --source-dir dataset/train/accident_raw \
     --output-dir dataset/train/accident \
     --dry-run
   ```
   This flags rows with a missing source file or a duration outside 2-5s.
4. Run it for real (drop `--dry-run`) to cut the clips with `ffmpeg`. Repeat
   for the validation split.

`ffmpeg` must be installed and on `PATH` (system dependency, not pip
installable).

## Training

```
python src/train_accident_model.py \
  --train-dir dataset/train --val-dir dataset/val \
  --epochs 20 --batch-size 2 \
  [--resume-from accident_model.pth]
```

Saves the best checkpoint (by validation accuracy) to `accident_model.pth`
(override with `--output`).

Class order used everywhere: index `0` = accident, index `1` = normal.

## Inference

```
python src/detect_accident_video.py
```

Loads `accident_model.pth` and runs a sliding 16-frame window classifier over
`test_videos/test_video.mp4`, flagging frames where accident probability
exceeds 0.35.
