import argparse  # for parsing CLI arguments
import csv  # for reading/writing attendance CSV
import json  # for JSON utilities (kept for potential future use)
import os  # for OS-level utilities (kept for potential future use)
from datetime import datetime  # for timestamps in attendance
from pathlib import Path  # for robust path handling
from urllib.request import urlretrieve  # for downloading model files

import cv2  # OpenCV for detection and recognition
import numpy as np  # numerical operations on embeddings

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}  # supported image formats
DATA_KNOWN = "data/known"  # default folder for labeled faces

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"  # YuNet detector model path
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"  # SFace recognizer model path
GALLERY_PATH = "models/gallery.npz"  # saved embeddings + labels
ATTENDANCE_PATH = "outputs/attendance.csv"  # attendance output CSV

DEFAULT_WIDTH = 1280  # default input width
DEFAULT_HEIGHT = 720  # default input height
DEFAULT_SCORE = 0.7  # detector score threshold
DEFAULT_NMS = 0.3  # detector NMS threshold
DEFAULT_TOPK = 5000  # detector top-k candidates
DEFAULT_COSINE = 0.36  # cosine similarity threshold

DOWNLOADS = {  # model filenames mapped to download URLs
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}


def ensure_models(model_dir: str):  # download missing models if needed
    Path(model_dir).mkdir(parents=True, exist_ok=True)  # ensure model directory exists
    for filename, urls in DOWNLOADS.items():  # iterate all required files
        path = Path(model_dir) / filename  # target file path
        if path.exists():  # skip if already present
            continue  # move to next file
        for url in urls:  # try each mirror
            try:  # attempt download
                print(f"Downloading {filename}...")  # user feedback
                urlretrieve(url, path)  # download to disk
                break  # stop after successful download
            except Exception:  # any download error
                if path.exists():  # remove partial file
                    path.unlink()  # delete incomplete download
        if not path.exists():  # if all mirrors failed
            raise RuntimeError(f"Failed to download {filename}. Check your connection or URLs.")  # fail fast


def create_detector(model_path: str, input_size, score_thr: float, nms_thr: float, top_k: int):  # build YuNet
    detector = cv2.FaceDetectorYN.create(  # create detector instance
        model_path,  # path to ONNX model
        "",  # optional config (unused)
        input_size,  # input image size
        score_thr,  # score threshold
        nms_thr,  # NMS threshold
        top_k,  # max candidates
    )  # end create call
    if detector is None:  # sanity check
        raise RuntimeError("Failed to create FaceDetectorYN.")  # fail if creation failed
    return detector  # return detector object


def create_recognizer(model_path: str):  # build SFace recognizer
    recognizer = cv2.FaceRecognizerSF.create(model_path, "")  # create recognizer instance
    if recognizer is None:  # sanity check
        raise RuntimeError("Failed to create FaceRecognizerSF.")  # fail if creation failed
    return recognizer  # return recognizer object


def detect_faces(detector, frame):  # detect all faces in a frame
    h, w = frame.shape[:2]  # image height/width
    detector.setInputSize((w, h))  # set detector input size
    _, faces = detector.detect(frame)  # run detection
    if faces is None or len(faces) == 0:  # no detections
        return []  # return empty list
    return faces  # return raw face detections


def pick_largest_face(faces):  # choose the biggest face by area
    if faces is None or len(faces) == 0:  # handle empty input
        return None  # nothing to pick
    return max(faces, key=lambda f: f[2] * f[3])  # select largest box


def extract_feature(recognizer, frame, face):  # extract embedding for a face
    aligned = recognizer.alignCrop(frame, face)  # align and crop face
    feature = recognizer.feature(aligned)  # compute embedding
    return feature.astype(np.float32)  # ensure float32


def cosine_similarity(query, gallery):  # compute cosine similarity scores
    query = query.reshape(1, -1)  # ensure 2D query
    gallery = gallery.reshape(gallery.shape[0], -1)  # flatten gallery
    query_norm = np.linalg.norm(query, axis=1, keepdims=True) + 1e-8  # avoid div by zero
    gallery_norm = np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8  # avoid div by zero
    return (query @ gallery.T) / (query_norm * gallery_norm.T)  # cosine similarity matrix


def ensure_attendance_file(attendance_path: str):  # create CSV if missing
    path = Path(attendance_path)  # normalize path
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
    if not path.exists():  # only create if absent
        with path.open("w", newline="", encoding="utf-8") as f:  # open file for write
            writer = csv.writer(f)  # CSV writer
            writer.writerow(["name", "date", "time", "similarity"])  # header row


def mark_attendance(attendance_path: str, name: str, similarity: float):  # append an attendance row
    now = datetime.now()  # current timestamp
    with Path(attendance_path).open("a", newline="", encoding="utf-8") as f:  # open for append
        writer = csv.writer(f)  # CSV writer
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), f"{similarity:.3f}"])  # row


def build_gallery(known_dir: str, detector, recognizer):  # build embeddings from known faces
    known_dir = Path(known_dir)  # normalize path
    if not known_dir.exists():  # validate existence
        raise RuntimeError(f"Known directory not found: {known_dir}")  # fail if missing

    embeddings = []  # list of face embeddings
    names = []  # list of corresponding labels

    person_names = sorted([p.name for p in known_dir.iterdir() if p.is_dir()])  # list person dirs
    if not person_names:  # no people found
        raise RuntimeError("No subfolders found in data/known. Add at least one person folder.")  # fail

    for name in person_names:  # iterate persons
        person_dir = known_dir / name  # path to person's images
        for file in person_dir.iterdir():  # iterate files
            if file.suffix.lower() not in IMAGE_EXTS:  # skip non-images
                continue  # go to next file
            image = cv2.imread(str(file))  # read image
            if image is None:  # skip unreadable files
                continue  # go to next file
            faces = detect_faces(detector, image)  # detect faces
            face = pick_largest_face(faces)  # choose largest face
            if face is None:  # no face found
                continue  # skip image
            feature = extract_feature(recognizer, image, face)  # compute embedding
            embeddings.append(feature)  # store embedding
            names.append(name)  # store label

    if not embeddings:  # if nothing collected
        raise RuntimeError("No faces detected in training images. Check lighting and image quality.")  # fail

    embeddings = np.vstack(embeddings).astype(np.float32)  # stack embeddings into array
    return embeddings, np.array(names, dtype=object)  # return embeddings and labels


def save_gallery(path: str, embeddings, names):  # save gallery to disk
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # ensure output dir exists
    np.savez_compressed(path, embeddings=embeddings, names=names)  # save as compressed NPZ


def load_gallery(path: str):  # load gallery from disk
    data = np.load(path, allow_pickle=True)  # load NPZ file
    return data["embeddings"].astype(np.float32), data["names"]  # return embeddings and names


def train(known_dir: str, detector_model: str, recognizer_model: str, gallery_path: str, width: int, height: int,  # train gallery
          score_thr: float, nms_thr: float, top_k: int):  # detector parameters
    ensure_models(Path(detector_model).parent)  # download models if needed
    detector = create_detector(detector_model, (width, height), score_thr, nms_thr, top_k)  # build detector
    recognizer = create_recognizer(recognizer_model)  # build recognizer
    embeddings, names = build_gallery(known_dir, detector, recognizer)  # build embeddings
    save_gallery(gallery_path, embeddings, names)  # persist gallery
    print(f"Gallery saved to {gallery_path} with {len(names)} samples")  # user feedback


def recognize_faces(frame, detector, recognizer, embeddings, names, cosine_threshold: float):  # recognize all faces
    faces = detect_faces(detector, frame)  # detect faces
    results = []  # list of (face, name, similarity)
    for face in faces:  # process each face
        feature = extract_feature(recognizer, frame, face)  # get embedding
        sims = cosine_similarity(feature, embeddings).flatten()  # similarity to gallery
        best_idx = int(np.argmax(sims))  # best match index
        best_sim = float(sims[best_idx])  # best similarity score
        name = names[best_idx]  # predicted label
        if best_sim < cosine_threshold:  # threshold check
            name = "Unknown"  # mark as unknown
        results.append((face, name, best_sim))  # store result
    return results  # return list of results


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
    ensure_models(Path(detector_model).parent)  # download models if missing
    embeddings, names = load_gallery(gallery_path)  # load known faces
    detector = create_detector(detector_model, (width, height), score_thr, nms_thr, top_k)  # detector
    recognizer = create_recognizer(recognizer_model)  # recognizer
    ensure_attendance_file(attendance_path)  # ensure CSV exists
    present = set()  # keep track of who is already marked

    cap = cv2.VideoCapture(camera_index)  # open webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # set height
    if not cap.isOpened():  # check webcam
        raise RuntimeError("Could not open webcam. Try a different --camera index.")  # fail if not opened

    print("Press 'q' to quit")  # instruction message
    while True:  # main loop
        ok, frame = cap.read()  # read a frame
        if not ok:  # stop if frame read failed
            break  # exit loop
        results = recognize_faces(frame, detector, recognizer, embeddings, names, cosine_threshold)  # recognize
        for (face, name, similarity) in results:  # iterate recognized faces
            if name == "Unknown":  # skip unknowns
                continue  # no drawing/attendance
            if name not in present:  # mark only once per run
                mark_attendance(attendance_path, name, similarity)  # write to CSV
                present.add(name)  # remember as present
            x, y, w, h = [int(v) for v in face[:4]]  # bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw box
            label = f"{name} ({similarity:.2f})"  # label string
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # draw label

        cv2.imshow("Face Recognition", frame)  # show the frame
        if cv2.waitKey(1) & 0xFF == ord("q"):  # quit on q
            break  # exit loop

    cap.release()  # release webcam
    cv2.destroyAllWindows()  # close windows


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
    ensure_models(Path(detector_model).parent)  # download models if missing
    embeddings, names = load_gallery(gallery_path)  # load known faces
    detector = create_detector(detector_model, (width, height), score_thr, nms_thr, top_k)  # detector
    recognizer = create_recognizer(recognizer_model)  # recognizer

    input_dir = Path(input_dir)  # normalize input path
    output_dir = Path(output_dir)  # normalize output path
    output_dir.mkdir(parents=True, exist_ok=True)  # ensure output dir exists

    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]  # list images
    if not image_files:  # no images found
        raise RuntimeError(f"No images found in {input_dir}")  # fail fast

    for file in image_files:  # process each image
        image = cv2.imread(str(file))  # read image
        if image is None:  # skip unreadable
            continue  # next file
        results = recognize_faces(image, detector, recognizer, embeddings, names, cosine_threshold)  # recognize
        for (face, name, similarity) in results:  # iterate results
            if name == "Unknown":  # skip unknowns
                continue  # no label for unknown
            x, y, w, h = [int(v) for v in face[:4]]  # bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw box
            label = f"{name} ({similarity:.2f})"  # label string
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # draw label
        out_path = output_dir / file.name  # output file path
        cv2.imwrite(str(out_path), image)  # save annotated image

    print(f"Processed {len(image_files)} images into {output_dir}")  # summary message


def build_parser():  # build CLI argument parser
    parser = argparse.ArgumentParser(description="Face recognition using OpenCV YuNet + SFace")  # base parser
    sub = parser.add_subparsers(dest="cmd", required=True)  # subcommands

    p_train = sub.add_parser("train", help="Build face gallery from data/known")  # train subcommand
    p_train.add_argument("--known", default=DATA_KNOWN)  # known folder
    p_train.add_argument("--detector", default=DETECTOR_MODEL)  # detector model path
    p_train.add_argument("--recognizer", default=RECOGNIZER_MODEL)  # recognizer model path
    p_train.add_argument("--gallery", default=GALLERY_PATH)  # output gallery path
    p_train.add_argument("--width", type=int, default=DEFAULT_WIDTH)  # input width
    p_train.add_argument("--height", type=int, default=DEFAULT_HEIGHT)  # input height
    p_train.add_argument("--score", type=float, default=DEFAULT_SCORE)  # detector score
    p_train.add_argument("--nms", type=float, default=DEFAULT_NMS)  # detector NMS
    p_train.add_argument("--topk", type=int, default=DEFAULT_TOPK)  # detector top-k

    p_webcam = sub.add_parser("webcam", help="Run live webcam recognition")  # webcam subcommand
    p_webcam.add_argument("--detector", default=DETECTOR_MODEL)  # detector model path
    p_webcam.add_argument("--recognizer", default=RECOGNIZER_MODEL)  # recognizer model path
    p_webcam.add_argument("--gallery", default=GALLERY_PATH)  # gallery path
    p_webcam.add_argument("--cosine", type=float, default=DEFAULT_COSINE)  # cosine threshold
    p_webcam.add_argument("--camera", type=int, default=0)  # webcam index
    p_webcam.add_argument("--attendance", default=ATTENDANCE_PATH)  # attendance CSV path
    p_webcam.add_argument("--width", type=int, default=DEFAULT_WIDTH)  # frame width
    p_webcam.add_argument("--height", type=int, default=DEFAULT_HEIGHT)  # frame height
    p_webcam.add_argument("--score", type=float, default=DEFAULT_SCORE)  # detector score
    p_webcam.add_argument("--nms", type=float, default=DEFAULT_NMS)  # detector NMS
    p_webcam.add_argument("--topk", type=int, default=DEFAULT_TOPK)  # detector top-k

    p_batch = sub.add_parser("batch", help="Run recognition on a folder of images")  # batch subcommand
    p_batch.add_argument("--detector", default=DETECTOR_MODEL)  # detector model path
    p_batch.add_argument("--recognizer", default=RECOGNIZER_MODEL)  # recognizer model path
    p_batch.add_argument("--gallery", default=GALLERY_PATH)  # gallery path
    p_batch.add_argument("--cosine", type=float, default=DEFAULT_COSINE)  # cosine threshold
    p_batch.add_argument("--input", default="data/unknown")  # input folder
    p_batch.add_argument("--output", default="outputs")  # output folder
    p_batch.add_argument("--width", type=int, default=DEFAULT_WIDTH)  # input width
    p_batch.add_argument("--height", type=int, default=DEFAULT_HEIGHT)  # input height
    p_batch.add_argument("--score", type=float, default=DEFAULT_SCORE)  # detector score
    p_batch.add_argument("--nms", type=float, default=DEFAULT_NMS)  # detector NMS
    p_batch.add_argument("--topk", type=int, default=DEFAULT_TOPK)  # detector top-k

    return parser  # return configured parser


def main():  # CLI entry point
    args = build_parser().parse_args()  # parse CLI arguments
    if args.cmd == "train":  # train command
        train(  # run training
            args.known,  # known folder
            args.detector,  # detector model
            args.recognizer,  # recognizer model
            args.gallery,  # gallery output path
            args.width,  # input width
            args.height,  # input height
            args.score,  # detector score
            args.nms,  # detector NMS
            args.topk,  # detector top-k
        )  # end train call
    elif args.cmd == "webcam":  # webcam command
        run_webcam(  # run webcam recognition
            args.detector,  # detector model
            args.recognizer,  # recognizer model
            args.gallery,  # gallery path
            args.cosine,  # cosine threshold
            args.camera,  # camera index
            args.attendance,  # attendance path
            args.width,  # frame width
            args.height,  # frame height
            args.score,  # detector score
            args.nms,  # detector NMS
            args.topk,  # detector top-k
        )  # end webcam call
    elif args.cmd == "batch":  # batch command
        run_batch(  # run batch recognition
            args.detector,  # detector model
            args.recognizer,  # recognizer model
            args.gallery,  # gallery path
            args.cosine,  # cosine threshold
            args.input,  # input directory
            args.output,  # output directory
            args.width,  # input width
            args.height,  # input height
            args.score,  # detector score
            args.nms,  # detector NMS
            args.topk,  # detector top-k
        )  # end batch call


if __name__ == "__main__":  # script entry check
    main()  # run main
