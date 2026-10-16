"""Train and use a text-classification baseline for fake-news detection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = ("fake", "real")
DEFAULT_MODEL_PATH = Path("models") / "fake_news_model.joblib"
MODEL_FORMAT_VERSION = 1

_LABEL_ALIASES = {
    "fake": "fake",
    "false": "fake",
    "0": "fake",
    "real": "real",
    "true": "real",
    "1": "real",
}
_LABEL_COLUMNS = ("label", "class", "target")
_TEXT_COLUMNS = ("title", "text")
_FALLBACK_TEXT_COLUMNS = ("content", "article", "news")


def normalize_label(value: str) -> str:
    """Map supported dataset labels to the canonical ``fake`` or ``real``."""
    normalized = value.strip().casefold()
    try:
        return _LABEL_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported label {value!r}; use fake/real, false/true, or 0/1 "
            "(0 means fake and 1 means real)."
        ) from error


def _resolve_columns(
    headers: list[str], requested: list[str] | None, *, purpose: str
) -> list[str]:
    by_normalized_name = {header.strip().casefold(): header for header in headers}
    if requested:
        missing = [
            name for name in requested if name.strip().casefold() not in by_normalized_name
        ]
        if missing:
            raise ValueError(
                f"Missing {purpose} column(s) {missing!r}. "
                f"Available columns: {headers!r}."
            )
        return [by_normalized_name[name.strip().casefold()] for name in requested]

    for candidates in (_TEXT_COLUMNS, _FALLBACK_TEXT_COLUMNS):
        selected = [
            by_normalized_name[name]
            for name in candidates
            if name in by_normalized_name
        ]
        if selected:
            return selected
    raise ValueError(
        "Could not infer text columns. Provide them with --text-columns. "
        f"Available columns: {headers!r}."
    )


def _read_csv(
    path: Path,
    *,
    text_columns: list[str] | None,
    label_column: str | None,
    assigned_label: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            headers = reader.fieldnames
            if not headers:
                raise ValueError(f"{path} has no CSV header.")
            selected_text_columns = _resolve_columns(
                headers, text_columns, purpose="text"
            )
            if assigned_label is None:
                normalized_headers = {
                    header.strip().casefold(): header for header in headers
                }
                selected_label = (
                    normalized_headers.get(label_column.strip().casefold())
                    if label_column
                    else next(
                        (
                            normalized_headers[name]
                            for name in _LABEL_COLUMNS
                            if name in normalized_headers
                        ),
                        None,
                    )
                )
                if selected_label is None:
                    raise ValueError(
                        f"Could not infer a label column in {path}. "
                        "Provide it with --label-column."
                    )
            else:
                selected_label = None

            texts: list[str] = []
            labels: list[str] = []
            for row in reader:
                text = " ".join(
                    (row.get(column) or "").strip()
                    for column in selected_text_columns
                ).strip()
                raw_label = assigned_label or (row.get(selected_label) or "")
                if not text and not raw_label:
                    continue
                if not text:
                    continue
                texts.append(text)
                labels.append(normalize_label(raw_label))
    except OSError as error:
        raise OSError(f"Could not read dataset file {path}: {error}") from error

    return texts, labels, selected_text_columns


def load_dataset(
    data_path: str | Path,
    *,
    text_columns: list[str] | None = None,
    label_column: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Load a labeled CSV or the conventional Fake.csv/True.csv dataset pair."""
    path = Path(data_path)
    if path.is_dir():
        csv_files = {child.name.casefold(): child for child in path.glob("*.csv")}
        fake_path = csv_files.get("fake.csv")
        real_path = csv_files.get("true.csv")
        if fake_path is None or real_path is None:
            raise ValueError(
                f"{path} must contain both Fake.csv and True.csv "
                "(the common Kaggle dataset format)."
            )
        fake_texts, fake_labels, selected = _read_csv(
            fake_path,
            text_columns=text_columns,
            label_column=None,
            assigned_label="fake",
        )
        real_texts, real_labels, _ = _read_csv(
            real_path,
            text_columns=text_columns,
            label_column=None,
            assigned_label="real",
        )
        texts = fake_texts + real_texts
        labels = fake_labels + real_labels
    else:
        texts, labels, selected = _read_csv(
            path,
            text_columns=text_columns,
            label_column=label_column,
        )

    if not texts:
        raise ValueError(f"No usable labeled text rows were found in {path}.")
    if set(labels) != set(LABELS):
        found = ", ".join(sorted(set(labels))) or "none"
        raise ValueError(
            f"Dataset must contain both fake and real examples; found: {found}."
        )
    return texts, labels, selected


def build_pipeline() -> Pipeline:
    """Create the text preprocessing and classification pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000)),
        ]
    )


def calculate_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    """Return standard binary-classification metrics with a fixed label order."""
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(
            precision_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=LABELS
        ).tolist(),
        "classification_report": classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def train_model(
    texts: list[str], labels: list[str], *, test_size: float, random_state: int
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit a pipeline and measure it on a stratified, held-out split."""
    if len(texts) != len(labels):
        raise ValueError("The number of texts and labels must match.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")
    if any(labels.count(label) < 2 for label in LABELS):
        raise ValueError("At least two examples of each class are required.")

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    metrics = calculate_metrics(y_test, pipeline.predict(x_test).tolist())
    metrics["train_examples"] = len(x_train)
    metrics["test_examples"] = len(x_test)
    return pipeline, metrics


def save_model(
    model_path: str | Path,
    pipeline: Pipeline,
    *,
    text_columns: list[str],
    label_column: str | None,
    metrics: dict[str, Any],
) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "pipeline": pipeline,
            "text_columns": text_columns,
            "label_column": label_column,
            "metrics": metrics,
        },
        path,
    )


def load_model(model_path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    if (
        not isinstance(artifact, dict)
        or artifact.get("format_version") != MODEL_FORMAT_VERSION
        or not isinstance(artifact.get("pipeline"), Pipeline)
    ):
        raise ValueError(f"{model_path} is not a supported fake-news model artifact.")
    return artifact


def predict_text(pipeline: Pipeline, text: str) -> dict[str, Any]:
    """Predict the class and class probabilities for one article."""
    probabilities = pipeline.predict_proba([text])[0]
    classifier = pipeline.named_steps["classifier"]
    scores = {
        str(label): float(probability)
        for label, probability in zip(classifier.classes_, probabilities)
    }
    label = str(classifier.classes_[probabilities.argmax()])
    return {
        "label": label,
        "confidence": scores[label],
        "probabilities": scores,
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a baseline fake-news text classifier."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train and evaluate on a held-out split")
    train.add_argument("--data", required=True, help="labeled CSV or Fake.csv/True.csv folder")
    train.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="output model artifact")
    train.add_argument(
        "--text-columns",
        nargs="+",
        help="text columns to combine (default: title and text, or an inferred content column)",
    )
    train.add_argument("--label-column", help="label column for a single labeled CSV")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="evaluate a saved model on labeled data")
    evaluate.add_argument("--data", required=True, help="labeled CSV or Fake.csv/True.csv folder")
    evaluate.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="model artifact")

    predict = commands.add_parser("predict", help="classify one article")
    predict.add_argument("--text", required=True, help="article text (or title and article together)")
    predict.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="model artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            texts, labels, selected_columns = load_dataset(
                args.data,
                text_columns=args.text_columns,
                label_column=args.label_column,
            )
            pipeline, metrics = train_model(
                texts,
                labels,
                test_size=args.test_size,
                random_state=args.random_state,
            )
            save_model(
                args.model,
                pipeline,
                text_columns=selected_columns,
                label_column=args.label_column,
                metrics=metrics,
            )
            _print_json({"model": args.model, "metrics": metrics})
        elif args.command == "evaluate":
            artifact = load_model(args.model)
            texts, labels, _ = load_dataset(
                args.data,
                text_columns=artifact["text_columns"],
                label_column=artifact["label_column"],
            )
            metrics = calculate_metrics(
                labels, artifact["pipeline"].predict(texts).tolist()
            )
            metrics["examples"] = len(texts)
            _print_json(metrics)
        else:
            artifact = load_model(args.model)
            _print_json(predict_text(artifact["pipeline"], args.text))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
