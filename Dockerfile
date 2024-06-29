FROM python:3.8


RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools wheel


COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt


COPY . /app

WORKDIR /app
EXPOSE 5767
ENV FLASK_APP=flaskapp.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5767
CMD ["flask", "run"]
