# Face Recognition (OpenCV YuNet + SFace)

## Install

python -m pip install -r requirements.txt

The first run will download two model files into `models/`.

## Data layout

Put labeled photos into subfolders under `data/known`:

- data/known/Alice/photo1.jpg
- data/known/Alice/photo2.jpg
- data/known/Bob/photo1.jpg

Put unlabeled photos into `data/unknown` for batch testing.

## Train (Build Gallery)

python main.py train

This saves:
- models/gallery.npz

## Live webcam (720p)

python main.py webcam

Attendance is logged to `outputs/attendance.csv` (only once per person per run).

To detect smaller/farther faces, use a lower `--score` and higher resolution:

python main.py webcam --width 1920 --height 1080 --score 0.6 --nms 0.3

## Batch recognition

python main.py batch --input data/unknown --output outputs

## Notes

- `--cosine` controls recognition strictness (higher = stricter). Default is `0.36`.
- If you get an error about `cv2.face`, ensure you installed `opencv-contrib-python`.
