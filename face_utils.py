from pathlib import Path
from datetime import datetime
import csv
import json
import re
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
MODELS = BASE / "models"
ATTENDANCE = BASE / "attendance"

MODEL_FILE = MODELS / "lbph_model.yml"
LABELS_FILE = MODELS / "labels.json"
CSV_FILE = ATTENDANCE / "attendance.csv"

CASCADE_FILE = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def ensure_dirs():
    for folder in (DATASET, MODELS, ATTENDANCE):
        folder.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        with CSV_FILE.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["student_id", "name", "date", "time"]
            )


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    if not value:
        raise ValueError("Invalid name or ID")
    return value


def get_detector():
    detector = cv2.CascadeClassifier(CASCADE_FILE)

    if detector.empty():
        raise RuntimeError("Face detector could not load")

    return detector


def capture_student(name, student_id, samples=30):
    ensure_dirs()

    name = safe_name(name)
    student_id = safe_name(student_id)

    folder = DATASET / f"{student_id}_{name}"
    folder.mkdir(parents=True, exist_ok=True)

    detector = get_detector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Webcam not found")

    count = 0

    try:
        while count < samples:
            success, frame = camera.read()

            if not success:
                break

            gray = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray, 1.2, 5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces[:1]:
                face = cv2.resize(
                    gray[y:y+h, x:x+w], (200, 200)
                )

                count += 1

                cv2.imwrite(
                    str(folder / f"{count:03d}.jpg"),
                    face
                )

                cv2.rectangle(
                    frame, (x, y), (x+w, y+h),
                    (0, 255, 0), 2
                )

            cv2.imshow("Student Registration", frame)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Captured {count} images")


def train_model():
    ensure_dirs()

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "Install opencv-contrib-python"
        )

    images = []
    labels = []
    label_map = {}

    for folder in sorted(DATASET.iterdir()):
        if not folder.is_dir():
            continue

        label = len(label_map)

        student_id, _, name = folder.name.partition("_")

        label_map[str(label)] = {
            "student_id": student_id,
            "name": name
        }

        for path in folder.glob("*.jpg"):
            image = cv2.imread(
                str(path), cv2.IMREAD_GRAYSCALE
            )

            if image is not None:
                images.append(
                    cv2.resize(image, (200, 200))
                )
                labels.append(label)

    if len(images) < 2:
        raise RuntimeError(
            "Register students before training"
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(
        images, np.array(labels, dtype=np.int32)
    )

    recognizer.write(str(MODEL_FILE))

    LABELS_FILE.write_text(
        json.dumps(label_map, indent=2)
    )

    print("Model training completed")


def mark_attendance(student_id, name):
    ensure_dirs()

    today = datetime.now().strftime("%Y-%m-%d")

    with CSV_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if (
            row["student_id"] == student_id
            and row["date"] == today
        ):
            return False

    now = datetime.now()

    with CSV_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        csv.writer(file).writerow([
            student_id,
            name,
            today,
            now.strftime("%H:%M:%S")
        ])

    return True


def recognize_and_mark(threshold=65):
    ensure_dirs()

    if not MODEL_FILE.exists():
        raise RuntimeError("Train the model first")

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "Install opencv-contrib-python"
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    labels = json.loads(
        LABELS_FILE.read_text()
    )

    detector = get_detector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Webcam not found")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            gray = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray, 1.2, 5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face = cv2.resize(
                    gray[y:y+h, x:x+w], (200, 200)
                )

                label, confidence = recognizer.predict(face)

                if (
                    str(label) in labels
                    and confidence <= threshold
                ):
                    person = labels[str(label)]

                    added = mark_attendance(
                        person["student_id"],
                        person["name"]
                    )

                    status = (
                        "Marked" if added
                        else "Already marked"
                    )

                    message = person["name"] + ": " + status
                    color = (0, 255, 0)

                else:
                    message = "Unknown Person"
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame, (x, y), (x+w, y+h),
                    color, 2
                )

                cv2.putText(
                    frame, message, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )

            cv2.imshow("Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
