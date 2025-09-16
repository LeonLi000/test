FROM python:3.10-slim

# Install system dependencies required by Jesse and scientific Python packages.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV JESSE_PROJECT=/app

CMD ["python", "backtest.py"]
