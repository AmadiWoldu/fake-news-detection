# Fake News Detection: Methodology and Evaluation Report

## Objective

Build a reproducible baseline that classifies news article text into the
dataset-provided `fake` and `real` labels. This is a text classification
experiment, not a system for independently establishing whether a claim is
true.

## Data

The experiment uses the [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download),
provided locally as `data/Fake.csv` and `data/True.csv`. In this paired format,
the file names provide the labels. The implementation also accepts a single
labeled CSV, where labels are read from
`label`, `class`, or `target` unless a column is specified explicitly. Text is
read from `title` and `text` by default where available. It does not bundle or
fetch a dataset; record the dataset name, version, license, and source when
conducting an actual experiment.

Before interpreting results, inspect class balance, missing values, duplicate
and near-duplicate articles, collection dates, source distributions, and label
quality. This repository's test fixtures are synthetic and are not evaluation
data.

## Preprocessing and feature selection

The scikit-learn pipeline applies lowercase conversion, Unicode accent
normalization, English stop-word filtering, and unigram/bigram TF-IDF with
sublinear term frequency. It caps the vocabulary at 100,000 features. The
vectorizer is fit only on the training split. Article title and body text are
included; subject, publisher, and date metadata are not included by default,
which helps avoid obvious metadata shortcuts.

## Model rationale

Logistic regression is a simple and interpretable-enough sparse linear
classification baseline. TF-IDF plus logistic regression is fast to train on
moderate text datasets and provides class probabilities for inspection. It
offers a useful baseline before considering more complex models, but these
probabilities are not calibrated by this project and must not be interpreted
as evidence verification.

## Training and evaluation

Training uses a stratified random train/test split with a 20% test fraction
and random seed 42. The notebook reports accuracy, macro-averaged
precision/recall/F1, a confusion matrix in `fake`, `real` order, and
per-class precision, recall, F1, and support. A separately collected dataset
can be evaluated with the `evaluate` command.

The notebook was run on the supplied local Kaggle-format CSVs: 44,898 usable
articles (23,481 fake and 21,417 real), with 35,918 training examples and 8,980
holdout examples. The exact dataset revision and license were not recorded.
Results below describe only this random holdout; duplicates, source artifacts,
and other dataset-specific signals may inflate them and limit generalization:

| Evaluation set | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Stratified holdout (seed 42, 20%) | 0.9919 | 0.9918 | 0.9919 | 0.9919 |
| Independent test set | Not measured | Not measured | Not measured | Not measured |

## Deployment

The model is available through the deployed web app at
[https://fake-news-detection-liab.onrender.com/](https://fake-news-detection-liab.onrender.com/).
The app accepts an article headline and body, sends their combined text to the
prediction service, and displays the predicted dataset label alongside the
model's estimated class probabilities. The service may take longer to respond
after a period of inactivity.

Deployment does not change the evaluation results above or establish
real-world fact-checking ability. The hosted prediction is still based on the
same dataset-dependent text classification approach and should be treated as
an exploratory signal, not evidence that an article is true or false.

## Challenges and limitations

The major methodological risk is learning dataset collection artifacts rather
than reliable indicators of misinformation. Duplicate stories across a random
split, publisher-specific writing styles, topic imbalance, and changing
language can inflate scores. Fake/real labels may also oversimplify misleading,
satirical, or partially true content. A stratified split alone does not solve
these problems.

For a stronger study, deduplicate before partitioning, reserve a
time-separated or source-separated test set, compare to a majority-class
baseline, report class counts and confidence intervals, inspect false
positives/negatives, and assess performance across relevant topics and groups.
For practical use, predictions require human review and corroboration from
reliable sources.

## Reproduction

Install `requirements.txt`, download and place the Kaggle dataset as described
in `README.md`, and run the cells in `fake_news_detector.ipynb` from top to
bottom. The notebook saves the fitted model and evaluation metadata to
`models/fake_news_model.joblib`, which can later be uploaded to Hugging Face.
For a command-line run or an independent test set, use:

```powershell
python fake_news_detector.py train --data .\data
python fake_news_detector.py evaluate --data .\data\independent_test.csv
python -m unittest discover -s tests -v
```
