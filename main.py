
import argparse
from face_utils import (
    ensure_dirs,
    capture_student,
    train_model,
    recognize_and_mark
)

def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Attendance System"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    enroll = sub.add_parser("enroll")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--student-id", required=True)

    sub.add_parser("train")
    sub.add_parser("attendance")

    args = parser.parse_args()
    ensure_dirs()

    if args.command == "enroll":
        capture_student(args.name, args.student_id)

    elif args.command == "train":
        train_model()

    elif args.command == "attendance":
        recognize_and_mark()


if __name__ == "__main__":
    main()
  
