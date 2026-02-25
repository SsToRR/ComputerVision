import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATA_KNOWN = "data/known"

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"
GALLERY_PATH = "models/gallery.npz"
ATTENDANCE_PATH = "outputs/attendance.csv"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_SCORE = 0.7
DEFAULT_NMS = 0.3
DEFAULT_TOPK = 5000
DEFAULT_COSINE = 0.36

DOWNLOADS = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}


def ensure_models(model_dir: str):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for filename, urls in DOWNLOADS.items():
        path = Path(model_dir) / filename
        if path.exists():
            continue
        for url in urls:
            try:
                print(f"Downloading {filename}...")
                urlretrieve(url, path)
                break
            except Exception:
                if path.exists():
                    path.unlink()
        if not path.exists():
            raise RuntimeError(f"Failed to download {filename}. Check your connection or URLs.")


def create_detector(model_path: str, input_size, score_thr: float, nms_thr: float, top_k: int):
    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        input_size,
        score_thr,
        nms_thr,
        top_k,
    )
    if detector is None:
        raise RuntimeError("Failed to create FaceDetectorYN.")
    return detector


def create_recognizer(model_path: str):
    recognizer = cv2.FaceRecognizerSF.create(model_path, "")
    if recognizer is None:
        raise RuntimeError("Failed to create FaceRecognizerSF.")
    return recognizer


def detect_faces(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return []
    return faces


def pick_largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def extract_feature(recognizer, frame, face):
    aligned = recognizer.alignCrop(frame, face)
    feature = recognizer.feature(aligned)
    return feature.astype(np.float32)


def cosine_similarity(query, gallery):
    query = query.reshape(1, -1)
    gallery = gallery.reshape(gallery.shape[0], -1)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
    gallery_norm = np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8
    return (query @ gallery.T) / (query_norm * gallery_norm.T)


def ensure_attendance_file(attendance_path: str):
    path = Path(attendance_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time", "similarity"])


def mark_attendance(attendance_path: str, name: str, similarity: float):
    now = datetime.now()
    with Path(attendance_path).open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), f"{similarity:.3f}"])


def build_gallery(known_dir: str, detector, recognizer):
    known_dir = Path(known_dir)
    if not known_dir.exists():
        raise RuntimeError(f"Known directory not found: {known_dir}")

    embeddings = []
    names = []

    person_names = sorted([p.name for p in known_dir.iterdir() if p.is_dir()])
    if not person_names:
        raise RuntimeError("No subfolders found in data/known. Add at least one person folder.")

    for name in person_names:
        person_dir = known_dir / name
        for file in person_dir.iterdir():
            if file.suffix.lower() not in IMAGE_EXTS:
                continue
            image = cv2.imread(str(file))
            if image is None:
                continue
            faces = detect_faces(detector, image)
            face = pick_largest_face(faces)
            if face is None:
                continue
            feature = extract_feature(recognizer, image, face)
            embeddings.append(feature)
            names.append(name)

    if not embeddings:
        raise RuntimeError("No faces detected in training images. Check lighting and image quality.")

    embeddings = np.vstack(embeddings).astype(np.float32)
    return embeddings, np.array(names, dtype=object)


def save_gallery(path: str, embeddings, names):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, names=names)


def load_gallery(path: str):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"].astype(np.float32), data["names"]


def train(known_dir: str, detector_model: str, recognizer_model: str, gallery_path: str, width: int, height: int,
          score_thr: float, nms_thr: float, top_k: int):
    ensure_models(Path(detector_model).parent)
    detector = create_detector(detector_model, (width, height), score_thr, nms_thr, top_k)
    recognizer = create_recognizer(recognizer_model)
    embeddings, names = build_gallery(known_dir, detector, recognizer)
    save_gallery(gallery_path, embeddings, names)
    print(f"Gallery saved to {gallery_path} with {len(names)} samples")


def recognize_faces(frame, detector, recognizer, embeddings, names, cosine_threshold: float):
    faces = detect_faces(detector, frame)
    results = []
    for face in faces:
        feature = extract_feature(recognizer, frame, face)
        sims = cosine_similarity(feature, embeddings).flatten()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        name = names[best_idx]
        if best_sim < cosine_threshold:
            name = "Unknown"
        results.append((face, name, best_sim))
    return results


def run_webcam(
    detector_model: str,
    recognizer_model: str,
    gallery_path: str,
    cosine_threshold: float,
    camera_index: int,
    attendance_path: str,
    width: int,
    height: int,
    score_thr: float,
    nms_thr: float,
    top_k: int,
):
    ensure_models(Path(detector_model).parent)
    embeddings, names = load_gallery(gallery_path)
    detector = create_detector(detector_model, (width, height), score_thr, nms_thr, top_k)
    recognizer = create_recognizer(recognizer_model)
    ensure_attendance_file(attendance_path)
    present = set()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera index.")

    print("Press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = recognize_faces(frame, detector, recognizer, embeddings, names, cosine_threshold)
        for (face, name, similarity) in results:
            if name == "Unknown":
                continue
            if name not in present:
                mark_attendance(attendance_path, name, similarity)
                present.add(name)
            x, y, w, h = [int(v) for v in face[:4]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{name} ({similarity:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_batch(
    detector_model: str,
    recognizer_model: str,
    gallery_path: str,
    cosine_threshold: float,
    input_dir: str,
    output_dir: str,
    width: int,
    height: int,
    score_thr: float,
    nms_thr: float,
    top_k: int,
):
    ensure_models(Path(detector_model).parent)
    embeddings, names = load_gallery(gallery_path)
    detector = create_detector(detector_model, (width, height), score_thr, nms_thr, top_k)
    recognizer = create_recognizer(recognizer_model)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not image_files:
        raise RuntimeError(f"No images found in {input_dir}")

    for file in image_files:
        image = cv2.imread(str(file))
        if image is None:
            continue
        results = recognize_faces(image, detector, recognizer, embeddings, names, cosine_threshold)
        for (face, name, similarity) in results:
            if name == "Unknown":
                continue
            x, y, w, h = [int(v) for v in face[:4]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{name} ({similarity:.2f})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_path = output_dir / file.name
        cv2.imwrite(str(out_path), image)

    print(f"Processed {len(image_files)} images into {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Face recognition using OpenCV YuNet + SFace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Build face gallery from data/known")
    p_train.add_argument("--known", default=DATA_KNOWN)
    p_train.add_argument("--detector", default=DETECTOR_MODEL)
    p_train.add_argument("--recognizer", default=RECOGNIZER_MODEL)
    p_train.add_argument("--gallery", default=GALLERY_PATH)
    p_train.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p_train.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p_train.add_argument("--score", type=float, default=DEFAULT_SCORE)
    p_train.add_argument("--nms", type=float, default=DEFAULT_NMS)
    p_train.add_argument("--topk", type=int, default=DEFAULT_TOPK)

    p_webcam = sub.add_parser("webcam", help="Run live webcam recognition")
    p_webcam.add_argument("--detector", default=DETECTOR_MODEL)
    p_webcam.add_argument("--recognizer", default=RECOGNIZER_MODEL)
    p_webcam.add_argument("--gallery", default=GALLERY_PATH)
    p_webcam.add_argument("--cosine", type=float, default=DEFAULT_COSINE)
    p_webcam.add_argument("--camera", type=int, default=0)
    p_webcam.add_argument("--attendance", default=ATTENDANCE_PATH)
    p_webcam.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p_webcam.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p_webcam.add_argument("--score", type=float, default=DEFAULT_SCORE)
    p_webcam.add_argument("--nms", type=float, default=DEFAULT_NMS)
    p_webcam.add_argument("--topk", type=int, default=DEFAULT_TOPK)

    p_batch = sub.add_parser("batch", help="Run recognition on a folder of images")
    p_batch.add_argument("--detector", default=DETECTOR_MODEL)
    p_batch.add_argument("--recognizer", default=RECOGNIZER_MODEL)
    p_batch.add_argument("--gallery", default=GALLERY_PATH)
    p_batch.add_argument("--cosine", type=float, default=DEFAULT_COSINE)
    p_batch.add_argument("--input", default="data/unknown")
    p_batch.add_argument("--output", default="outputs")
    p_batch.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p_batch.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p_batch.add_argument("--score", type=float, default=DEFAULT_SCORE)
    p_batch.add_argument("--nms", type=float, default=DEFAULT_NMS)
    p_batch.add_argument("--topk", type=int, default=DEFAULT_TOPK)

    return parser


def main():
    args = build_parser().parse_args()
    if args.cmd == "train":
        train(
            args.known,
            args.detector,
            args.recognizer,
            args.gallery,
            args.width,
            args.height,
            args.score,
            args.nms,
            args.topk,
        )
    elif args.cmd == "webcam":
        run_webcam(
            args.detector,
            args.recognizer,
            args.gallery,
            args.cosine,
            args.camera,
            args.attendance,
            args.width,
            args.height,
            args.score,
            args.nms,
            args.topk,
        )
    elif args.cmd == "batch":
        run_batch(
            args.detector,
            args.recognizer,
            args.gallery,
            args.cosine,
            args.input,
            args.output,
            args.width,
            args.height,
            args.score,
            args.nms,
            args.topk,
        )


if __name__ == "__main__":
    main()
