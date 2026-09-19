# Author: Vishesh
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .dataset import discover_sequences, load_frames
from .demo import generate_demo_dataset
from .evaluation import evaluate
from .experiment import run_experiment
from .features import extract_features
from .model import ActionRecognizer
from .reporting import write_annotated_video, write_preview_frame
from .validation import validate_dataset


def _get_labeled_records(dataset: Path) -> list:
    records = []
    all_records = discover_sequences(dataset)

    for record in all_records:
        if record.annotation is not None:
            records.append(record)

    return records


def _metrics_summary(metrics: dict) -> dict:
    summary = {}

    for key, value in metrics.items():
        if key != "results":
            summary[key] = value

    return summary


def _train(dataset: Path, model_path: Path) -> tuple[ActionRecognizer, int]:
    records = _get_labeled_records(dataset)

    if len(records) < 6:
        raise ValueError("Training needs at least six labeled 40-frame sequence folders")

    features, labels = [], []

    for index, record in enumerate(records, 1):
        frames = load_frames(record)
        vector, _ = extract_features(frames)
        features.append(vector)

        annotation = record.annotation
        if annotation is not None:
            labels.append(annotation.class_id)

        print(f"[{index:>3}/{len(records)}] features: {record.name}")

    model = ActionRecognizer().fit_features(np.vstack(features), np.asarray(labels))
    model.save(model_path)

    return model, len(records)


def command_train(args: argparse.Namespace) -> None:
    _, count = _train(Path(args.dataset), Path(args.model))
    print(f"Saved {args.model} ({count} training sequences)")


def command_predict(args: argparse.Namespace) -> None:
    frames = load_frames(args.sequence)
    prediction = ActionRecognizer.load(args.model).predict(frames)
    output = Path(args.output)

    write_annotated_video(frames, prediction, output / "prediction.mp4")
    write_preview_frame(frames, prediction, output / "median_frame.png")

    result = {
        "label": prediction.label,
        "class_id": prediction.class_id,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
    }

    (output / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    records = discover_sequences(args.dataset)
    metrics = evaluate(ActionRecognizer.load(args.model), records, args.output)
    print(json.dumps(_metrics_summary(metrics), indent=2))


def command_validate(args: argparse.Namespace) -> None:
    report = validate_dataset(discover_sequences(args.dataset), strict_proposal=not args.allow_nonstandard)
    print(json.dumps(report.as_dict(), indent=2))
    if not report.valid:
        raise SystemExit(2)


def command_experiment(args: argparse.Namespace) -> None:
    result = run_experiment(
        args.dataset,
        args.output,
        seed=args.seed,
        strict_proposal=not args.allow_nonstandard,
        qualitative_videos=not args.no_videos,
    )

    summary = {
        "model": str(result.model_path),
        "train_sequences": result.train_count,
        "test_sequences": result.test_count,
        "metrics": _metrics_summary(result.metrics),
    }

    print(json.dumps(summary, indent=2))


def command_demo(args: argparse.Namespace) -> None:
    output = Path(args.output)
    data = output / "data"

    if args.fresh and output.exists():
        shutil.rmtree(output)

    generate_demo_dataset(data, per_class=8)
    train_root = output / "train"
    test_root = output / "test"

    for action_dir in sorted(data.iterdir()):
        sequences = sorted(action_dir.iterdir())

        for sequence in sequences[:6]:
            destination = train_root / action_dir.name / sequence.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(sequence, destination, dirs_exist_ok=True)

        for sequence in sequences[6:]:
            destination = test_root / action_dir.name / sequence.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(sequence, destination, dirs_exist_ok=True)

    model_path = output / "action_model.joblib"
    model, _ = _train(train_root, model_path)

    metrics = evaluate(model, discover_sequences(test_root), output / "evaluation")
    sample = discover_sequences(test_root)[0]
    frames = load_frames(sample)
    prediction = model.predict(frames)

    write_annotated_video(frames, prediction, output / "sample_prediction.mp4")
    write_preview_frame(frames, prediction, output / "sample_prediction.png")

    print(json.dumps(_metrics_summary(metrics), indent=2))
    print(f"Demo artifacts written to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ultra-vision", description="Ultra Vision computer vision final project")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train from labeled sequence folders")
    train.add_argument("dataset")
    train.add_argument("--model", default="artifacts/action_model.joblib")
    train.set_defaults(func=command_train)

    predict = commands.add_parser("predict", help="Classify and annotate one sequence folder")
    predict.add_argument("sequence")
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", default="artifacts/prediction")
    predict.set_defaults(func=command_predict)

    test = commands.add_parser("evaluate", help="Compute accuracy, F1, confusion matrix and median-frame IoU")
    test.add_argument("dataset")
    test.add_argument("--model", required=True)
    test.add_argument("--output", default="artifacts/evaluation")
    test.set_defaults(func=command_evaluate)

    validate = commands.add_parser("validate", help="Validate dataset against the 72-sequence proposal contract")
    validate.add_argument("dataset")
    validate.add_argument(
        "--allow-nonstandard",
        action="store_true",
        help="Allow datasets other than the exact 72-sequence design",
    )
    validate.set_defaults(func=command_validate)

    experiment = commands.add_parser(
        "experiment",
        help="Run validation, stratified training, evaluation and artifact generation",
    )
    experiment.add_argument("dataset")
    experiment.add_argument("--output", default="artifacts/experiment")
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--allow-nonstandard", action="store_true")
    experiment.add_argument("--no-videos", action="store_true")
    experiment.set_defaults(func=command_experiment)

    demo = commands.add_parser("demo", help="Generate data and exercise the full pipeline")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--fresh", action="store_true")
    demo.set_defaults(func=command_demo)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
