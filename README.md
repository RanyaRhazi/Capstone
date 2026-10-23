# KNN Digit Classifier API (Flask + Docker)

A K-Nearest Neighbors model that classifies 8x8 images of handwritten digits (0–9), served through a Flask REST API and packaged with Docker. Built for a machine learning deployment course assignment. The original write-up is in [`KNN_Dep.pdf`](KNN_Dep.pdf).

## What it does

- **Data:** scikit-learn's [digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset): 1,797 grayscale 8x8 images, pixel values 0–16.
- **Model:** `KNeighborsClassifier(n_neighbors=3)` on MinMax-scaled pixels, with a 75/25 train/test split (`random_state=0`).
- **Serving:** `POST /predict` takes one image and returns the predicted digit. The server applies the same scaler used in training.

## Results

| Metric | Value |
|---|---|
| Test accuracy (450 held-out images) | 0.9889 |

This is the output of `python train.py`, and matches the notebook ([`KNN_Model.ipynb`](KNN_Model.ipynb)).

## Run it

### With Docker

```bash
docker build -t knn-digits .
docker run -p 5767:5767 knn-digits
```

The image trains the model during the build, so no model files need to be committed.

### Locally

```bash
pip install -r requirements.txt
python train.py          # writes knn_model.pkl and scaler.pkl
python flaskapp.py       # serves on http://localhost:5767
```

## API

### `POST /predict`

Send the raw pixel values (0–16) of one image, either as a flat list of 64 numbers or as 8 rows of 8. Don't scale them yourself; the server does that.

```bash
curl -X POST http://localhost:5767/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]}'
```

Response:

```json
{"prediction": [0]}
```

Invalid input (missing `image`, wrong number of values, non-numeric values) returns HTTP 400 with an `error` message.

### `GET /health`

Returns `{"status": "ok"}`.

## Tests

```bash
python train.py
python -m unittest -v
```

The tests cover correct predictions on known digits, both input shapes, and the 400 error cases.

## Project files

| File | Purpose |
|---|---|
| `train.py` | Trains the model and saves `knn_model.pkl` and `scaler.pkl` |
| `flaskapp.py` | Flask API |
| `test_api.py` | API tests |
| `KNN_Model.ipynb` | Original training notebook |
| `KNN_Dep.pdf` | Original deployment report |
| `Dockerfile` | Container image |

## Limitations

- The digits dataset is small and clean, so 98.9% accuracy here says little about real handwriting. Images must already be 8x8 and on the 0–16 scale.
- KNN keeps the whole training set in memory and compares every request against it. That's fine at this size but doesn't scale to large datasets.
- `flask run` is Flask's development server. A production deployment would sit behind a WSGI server such as Gunicorn.
- The notebook's second cell builds an example request from already-scaled test data. The API scales its input again, so send raw pixel values as shown above instead.
