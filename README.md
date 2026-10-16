# Fake News Detection

A reproducible, classical NLP baseline that classifies article text as `fake` or
`real`. It uses TF-IDF features and logistic regression from scikit-learn. It is
intended for experimentation and education, not as an automated fact-checker:
predictions are dataset-dependent and must not be treated as proof that a story
is true or false.

## Live demo

Try the deployed web app at
[fake-news-detection-liab.onrender.com](https://fake-news-detection-liab.onrender.com/).
Enter an article headline and text, then select **Analyze article** to see the
model's predicted label and estimated class probabilities. The hosted service
may take a little while to respond when waking from inactivity. As with the
local app, the result reflects patterns learned from the training dataset; it
does not verify an article's claims or sources.

## Main workflow

Use [fake_news_detector.ipynb](./fake_news_detector.ipynb) as the primary
workflow to inspect the supplied data, train and evaluate the model, save the
model artifact, and try a prediction. The notebook uses the shared functions in
`fake_news_detector.py`; the command-line interface remains available for
repeatable training, evaluation, and prediction outside the notebook.

## Requirements

- Python 3.10 or newer
- The [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download)
- The VS Code Jupyter extension (or Jupyter) to run the notebook

Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In VS Code, open `fake_news_detector.ipynb`, select the environment where these
requirements are installed as the kernel, and run the cells from top to bottom.

## Prepare data

Download the Kaggle dataset and place its CSV files in the project's `data/`
directory. The project does not silently download data. The expected layout is:

```text
data\
  Fake.csv
  True.csv
```

Those files should have a `text` column; an optional `title` column is combined
with it. Rows in `Fake.csv` are labeled fake and rows in `True.csv` real.

A single labeled CSV is also supported. By default, the text column is selected
from `title` and `text` (when present), otherwise from `content`, `article`, or
`news`. The label column is inferred from `label`, `class`, or `target`.
Recognized labels are `fake`/`false`/`0` and `real`/`true`/`1`; for numeric
labels, 0 means fake and 1 means real. Explicit columns can be selected with
`--text-columns` and `--label-column`.

## Train and evaluate

Open `fake_news_detector.ipynb` in VS Code or Jupyter and run its cells from
top to bottom. It reports class counts and holdout metrics, and saves the
complete trained pipeline to `models\fake_news_model.joblib`. The artifact is
ready to upload to Hugging Face later.

The optional CLI workflow, from the project directory, is:

```powershell
python fake_news_detector.py train --data .\data
```

The command makes a stratified train/test split (20% test by default), fits the
vectorizer only on the training partition, reports accuracy, macro precision,
macro recall, macro F1, a confusion matrix, and a per-class report. It saves the
trained pipeline and split metrics to
`models\fake_news_model.joblib`. Override the split, seed, output path, or CSV
columns with:

```powershell
python fake_news_detector.py train --data .\data\articles.csv `
  --text-columns headline article_body --label-column verdict `
  --test-size 0.25 --random-state 42 --model .\models\baseline.joblib
```

To evaluate a saved model against a separate labeled dataset with the same
column names:

```powershell
python fake_news_detector.py evaluate --data .\data\held_out.csv `
  --model .\models\fake_news_model.joblib
```

For one article, pass the title and body together if both are available:

```powershell
python fake_news_detector.py predict --model .\models\fake_news_model.joblib `
  --text "The article headline and full article text go here."
```

Prediction output is JSON containing the predicted label, the model's estimated
confidence, and probabilities for each class. Confidence is not a calibrated
measure of truthfulness.

## Method

The pipeline lowercases and accent-normalizes text, removes English stop words,
and extracts unigram and bigram TF-IDF features. Logistic regression classifies
the resulting sparse vectors. Fitting vectorization inside the scikit-learn
pipeline prevents test vocabulary from leaking into training. A fixed random
seed and stratified split make the baseline repeatable.

Only article/title text is used by default. Dataset metadata such as date and
subject is excluded to reduce shortcut learning and dataset-specific leakage.
Text artifacts can still encode publisher, formatting, and collection-source
cues, so strong held-out scores do not establish real-world generalization.

## Tests

Run the local tests (they use temporary synthetic fixtures and do not require a
downloaded dataset):

```powershell
python -m unittest discover -s tests -v
```

## Limitations and responsible use

- A model learns patterns and labeling conventions in its training dataset; it
  does not verify claims against evidence or sources.
- Dataset imbalances, duplicates, temporal shifts, political or domain bias,
  and source leakage can make evaluation misleading.
- Use a genuinely independent, preferably time-separated test set for stronger
  evaluation. Deduplicate articles before splitting if near-duplicates exist.
- Do not use a prediction alone to label people, suppress reporting, or make
  consequential decisions. Human review and corroborating evidence are needed.
- Only load model artifacts you created or otherwise trust: joblib artifacts
  use Python serialization and can execute code when loaded.
