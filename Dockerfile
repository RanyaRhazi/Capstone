FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train at build time so the image always contains knn_model.pkl and scaler.pkl.
RUN python train.py

ENV FLASK_APP=flaskapp.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5767
EXPOSE 5767
CMD ["flask", "run"]
