import csv
import tempfile
import unittest
from pathlib import Path

from fake_news_detector import (
    calculate_metrics,
    load_dataset,
    load_model,
    normalize_label,
    predict_text,
    save_model,
    train_model,
)


class DatasetTests(unittest.TestCase):
    def test_normalizes_supported_labels(self):
        self.assertEqual(normalize_label(" FAKE "), "fake")
        self.assertEqual(normalize_label("False"), "fake")
        self.assertEqual(normalize_label("1"), "real")

    def test_loads_kaggle_fake_true_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for filename, text in (("Fake.csv", "fabricated claim"), ("True.csv", "verified report")):
                with (directory / filename).open("w", encoding="utf-8", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=["title", "text"])
                    writer.writeheader()
                    writer.writerow({"title": "headline", "text": text})

            texts, labels, columns = load_dataset(directory)

        self.assertEqual(texts, ["headline fabricated claim", "headline verified report"])
        self.assertEqual(labels, ["fake", "real"])
        self.assertEqual(columns, ["title", "text"])

    def test_loads_single_labeled_csv_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "articles.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["Content", "Class"])
                writer.writeheader()
                writer.writerow({"Content": "made up article", "Class": "fake"})
                writer.writerow({"Content": "reported article", "Class": "real"})

            texts, labels, columns = load_dataset(path)

        self.assertEqual(texts, ["made up article", "reported article"])
        self.assertEqual(labels, ["fake", "real"])
        self.assertEqual(columns, ["Content"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "fabricated shocking rumor fake story",
            "invented false claim sensational rumor",
            "fake fabricated conspiracy story",
            "unverified made up shocking report",
            "false rumor invented headline",
            "sensational fabricated news claim",
            "verified official report confirmed facts",
            "reliable journalist reported confirmed facts",
            "official source confirms accurate report",
            "verified factual news from sources",
            "confirmed accurate journalist report",
            "reliable verified official news",
        ]
        self.labels = ["fake"] * 6 + ["real"] * 6

    def test_training_metrics_and_prediction(self):
        pipeline, metrics = train_model(
            self.texts, self.labels, test_size=0.33, random_state=42
        )

        self.assertEqual(metrics["train_examples"] + metrics["test_examples"], 12)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertIn(predict_text(pipeline, "fabricated false rumor")["label"], {"fake", "real"})
        self.assertEqual(
            calculate_metrics(["fake"], ["fake"])["confusion_matrix"],
            [[1, 0], [0, 0]],
        )

    def test_saved_model_can_be_loaded(self):
        pipeline, metrics = train_model(
            self.texts, self.labels, test_size=0.33, random_state=42
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.joblib"
            save_model(
                path,
                pipeline,
                text_columns=["title", "text"],
                label_column=None,
                metrics=metrics,
            )

            artifact = load_model(path)

        self.assertEqual(artifact["text_columns"], ["title", "text"])
        self.assertIn(predict_text(artifact["pipeline"], "verified official facts")["label"], {"fake", "real"})


if __name__ == "__main__":
    unittest.main()
