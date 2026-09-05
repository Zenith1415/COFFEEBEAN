FROM python:3.12-slim AS mlflow
# MLflow 3.15.2's Starlette WSGI adapter imports ``anyio.from_thread``.
# AnyIO 4.15 switched top-level modules to lazy imports and breaks that path.
RUN pip install --no-cache-dir anyio==4.14.0 mlflow==3.15.2 psycopg2-binary==2.9.10

FROM apache/airflow:3.3.1-python3.12 AS airflow
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends autoconf automake build-essential ca-certificates libtool pkg-config \
    && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir mlflow==3.15.2 numpy==1.26.4 psycopg2-binary==2.9.10 torch==2.5.1 tqdm==4.67.1
