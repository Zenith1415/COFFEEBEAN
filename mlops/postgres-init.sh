#!/bin/sh
set -eu

psql --set ON_ERROR_STOP=1 \
  --set airflow_password="$AIRFLOW_DB_PASSWORD" \
  --set mlflow_password="$MLFLOW_DB_PASSWORD" \
  --username "$POSTGRES_USER" <<-SQL
  CREATE USER airflow WITH PASSWORD :'airflow_password';
  CREATE DATABASE airflow OWNER airflow;
  CREATE USER mlflow WITH PASSWORD :'mlflow_password';
  CREATE DATABASE mlflow OWNER mlflow;
SQL
