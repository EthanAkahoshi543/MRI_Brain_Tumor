# MRI VisionGuard

A web app that looks at a brain MRI image and says whether it shows a tumour. It also shows which part of the scan the decision was based on, and it says how far the answer can be trusted.

> **Research and teaching project.** It is not a medical device and must not be used to diagnose or treat anyone.

![The app's start page: sample scans, difficult inputs and the drop zone](docs/screenshots/overview.png)

## What it does

You upload one image from a brain MRI scan. The app then:

1. **Checks the image quality.** It runs nine checks, for example whether the image is blurry, too dark or badly cropped.
2. **Makes a prediction.** A small AI model, trained from scratch for this project, gives a tumour score from 0 to 1. A score of 0.5 or higher counts as a tumour.
3. **Shows where it looked.** A heatmap is drawn over the scan to show which areas pushed the model towards its answer. The strongest area appears first.
4. **Checks its own answer.** If the image quality is borderline, the app tests a slightly changed copy of the scan (more contrast, a tighter crop or a mirror image) to see whether the answer changes. Each result is labelled consistent, sensitive, poor quality or needs review, and the app lists each step that led to that label. This check never changes the prediction.

![A result: heatmap, prediction and the steps behind it](docs/screenshots/result.png)

![Four frames of the heatmap building up over about 1.5 seconds](docs/screenshots/heatmap-reveal.png)

### Ask about this result

Under each result there is an optional chat assistant, powered by Claude (Anthropic's AI model). It explains the result in everyday language and answers follow-up questions. For people who have been diagnosed by a doctor, or who care for someone who has, it can also talk through usual next steps and help find counselling and support groups nearby.

It was built with safety rules:

- It sees only the numbers from the result, never the image.
- It never treats the app's prediction as a diagnosis and always sends people to their doctor about their own health.
- It searches only trusted health, cancer and crisis-support websites, and links to the sources it used.
- If someone sounds in distress, it gives crisis lines first (988 and 911 in the US, Samaritans in the UK).

## How well it works

The model was tested on scans it never saw during training.

| Test set | Scans | Tumours caught | Healthy scans correctly cleared |
|---|---|---|---|
| Main test set | 599 | 93.6% | 93.9% |
| Public benchmark set | 1,600 | 92.9% | 87.5% |
| Held-out set | 253 | 73.5% | 95.9% |

The held-out set comes from the same original source as part of the training data, so it is not a test on scans from a different hospital. It shows the model's main weakness: it misses about one tumour in four there.

To test on scans from a different source, the model was also run on 1,743 images of healthy brains from three London hospitals (the [IXI dataset](https://brain-development.org/ixi-dataset/), CC BY-SA 3.0). None of these were used in training.

- 95.6% were correctly cleared as healthy.
- The self-check flagged 46% of the wrong answers for a human to review, but only 3.4% of the right ones. A wrong answer was about 13 times more likely to be flagged.

This set has no tumour scans, so it can't show how well the model finds tumours in scans from new hospitals.

Full figures, including accuracy, F1 and ROC-AUC, are in [`docs/evaluation.md`](docs/evaluation.md).

## Finding and fixing a hidden shortcut

An earlier version of the model scored well, but it was cheating without anyone meaning it to. In one of the training datasets, the tumour images were large and dark and the healthy images were small and bright. The model learned to tell them apart by size and brightness instead of looking at the brain. Its heatmaps lit up the edge of the skull and the image border, not the tumours.

The test data had problems too. About 97% of the old "external" test set turned out to be copies of training images, and near-copies between the training and test sets made the scores look better than they were.

Fixing this took three changes:

- **Separating the data properly.** Near-copies are grouped together and kept on one side of the split, and any training image that matches a test image is removed. An automated test checks this.
- **Hiding the giveaway clues.** Every image is cropped to the head, made square, turned greyscale, brightness-balanced and resized to the same size, so image size and brightness no longer give the answer away.
- **Balanced training.** The model sees an even mix of each dataset and each class, so it can't use where an image came from to guess the answer.

One gap remains. The datasets don't say which images come from the same patient, so images of one person could still end up in both training and testing.

## Limitations

- It was trained on a few thousand public MRI images, without patient information or any clinical testing.
- It does worse on scans from new sources and misses some tumours. A "No tumour" result never rules one out.
- The scores are not probabilities. A 99% score does not mean a 99% chance of a tumour.
- It gives an answer for any image, even one that isn't a brain MRI.
- The heatmap is rough. It shows roughly where the model looked, not the outline of a tumour.

## Built with

Python, PyTorch (the AI model), OpenCV (image checks), FastAPI (the web server), and plain HTML, CSS and JavaScript for the web page. The assistant uses the Anthropic Python SDK.

The model is small: 393,618 parameters and a 1.6 MB file. It runs on an ordinary CPU in about 2 ms per image, or about 16 ms including the quality checks and heatmap. It uses a Grad-CAM heatmap taken from the fourth of its five convolution blocks.

## For developers

### Running it

You need Python 3.13. No GPU is required.

```bash
python -m venv .venv
.venv\Scripts\pip install -r webapp/requirements.txt      # Windows
# source .venv/bin/activate && pip install -r webapp/requirements.txt   (macOS/Linux)

.venv\Scripts\python -m uvicorn webapp.app:app
```

Then open http://127.0.0.1:8000 and try one of the sample scans.

To turn on the assistant, set an Anthropic API key before starting the server (`set ANTHROPIC_API_KEY=...` on Windows, `export ANTHROPIC_API_KEY=...` elsewhere). Without one, everything else works and the assistant says it isn't set up.

Uploads are limited to JPEG or PNG, 10 MB and 50 megapixels. Images are processed in memory and never saved to disk.

### Tests

```bash
.venv\Scripts\pip install pytest httpx
.venv\Scripts\python -m pytest
```

The 53 tests cover image preparation, the quality checks, the reliability rules, the API, the assistant (with a stand-in client, so no API key is needed), the data split, and a check that the app, `mainTest.py` and a direct model call give the same answer.

### API

| Method | Path | Returns |
|---|---|---|
| `POST` | `/api/predict` | Prediction, scores, heatmap, quality checks, reliability status and decision trace for an uploaded image (`file` form field) |
| `POST` | `/api/ask` | The assistant's answer, given a result from `/api/predict` and the conversation so far (JSON: `result`, `messages`) |
| `GET` | `/api/model` | Class names, input size, parameter count and the quality and reliability thresholds |
| `GET` | `/api/health` | `{"status": "ok"}` |

### Project layout

```
model.py               Model definitions, preprocessing and loading
gradcam.py             Grad-CAM and heatmap colouring
quality_analysis.py    The nine image quality checks
reliability.py         Follow-up rules, status and decision trace
mainTest.py            Predict on one image from the command line
webapp/                FastAPI app and the static front end (HTML, CSS, JS)
tests/                 pytest suite
docs/                  Evaluation results, IXI results, app flow, screenshots
experiments/           Investigation, data split, training and evaluation scripts
models/                Training run summaries (thresholds and validation scores)
```

The training data is not in the repo. It came from two public Kaggle brain MRI collections plus IXI, and goes in `datasets/` (ignored by git). The trainer is [`experiments/engineer/train.py`](experiments/engineer/train.py). `mainTrain.py` is the original training script, kept so the earlier investigation can be reproduced.
