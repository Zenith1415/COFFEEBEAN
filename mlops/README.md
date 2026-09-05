# RNNoise retraining

This setup retrains the exact RNNoise v0.2 architecture already used by
COFFEEBEAN. It pins and verifies the upstream source, records input and feature
hashes, tracks the run in MLflow, requires a separate evaluation corpus, gates
the candidate, and exports quantized `rnnoise_data.c/.h`. Airflow schedules and
observes the job; it does not hide the training command.

## Data contract

Prepare four headerless, mono, native-endian signed-int16 PCM files at 48 kHz:

```text
mlops/data/input/train-speech.pcm
mlops/data/input/train-noise.pcm
mlops/data/input/eval-speech.pcm
mlops/data/input/eval-noise.pcm
```

Keep speakers and source recordings disjoint between train and evaluation.
Complete `DATA_CARD.example.md`, save it as `mlops/data/input/DATA_CARD.md`, and
record every source, version, license, transformation and split rule. Do not
commit audio, features, credentials, checkpoints, or Airflow logs.

Upstream recommends at least 10,000 feature sequences and 200,000 or more for a
serious run. The Airflow default is 200,000 train sequences. Each sequence is
about 784 KB, so budget roughly 157 GB for the generated training features.

## Start local orchestration

```bash
cd mlops
cp .env.example .env
# Replace every value in .env. Keep database passwords URL-safe and generate
# AIRFLOW_FERNET_KEY with: python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
mkdir -p data/input data/cache/rnnoise logs
docker compose up --build -d
```

Open Airflow at <http://localhost:8080> and MLflow at
<http://localhost:5000>. Airflow 3 Simple Auth stores the generated admin
password in `logs/simple_auth_manager_passwords.json.generated`. Trigger
`rnnoise_retraining` after validating its parameters. The DAG is intentionally
manual: enable a schedule only after a versioned data-ingestion or drift signal
exists.

For a cheap preflight before allocating training compute:

```bash
docker compose run --rm airflow-scheduler python \
  /opt/coffeebean/mlops/rnnoise_pipeline.py validate \
  --train-speech /data/input/train-speech.pcm \
  --train-noise /data/input/train-noise.pcm \
  --eval-speech /data/input/eval-speech.pcm \
  --eval-noise /data/input/eval-noise.pcm \
  --data-card /data/input/DATA_CARD.md \
  --output /data/preflight-manifest.json
```

The default epoch count targets about 75,000 optimizer updates, matching the
upstream recommendation. Set `max_validation_loss` only after establishing a
representative baseline; set `baseline_validation_loss` to require a strict
improvement. Without either, finite held-out loss creates a candidate, not a
production promotion.

## Promotion checklist

The pipeline never auto-deploys. Before replacing the checked-in runtime model:

1. Review the MLflow lineage, system metrics, train/validation loss and data card.
2. Run the existing offline SI-SDR/STOI benchmark on fixed clean/noisy WAV pairs.
3. Listen to speech, silence, non-speech and impulsive-noise cases for artifacts.
4. Compile the exported C weights and pass the 60-second edge real-time gate.
5. Approve and copy the two C files into a reviewed RNNoise source build.

For a shared or production service, move PostgreSQL and artifacts to managed
backups/object storage, put MLflow and Airflow behind TLS/SSO, use a secrets
manager, and run training in an isolated GPU job rather than LocalExecutor.

## Design references

Verified 2026-09-05 against the primary documentation:

- [Xiph RNNoise training and export procedure](https://github.com/xiph/rnnoise#training)
- [MLflow tracking concepts](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow tracking-server architecture](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [Airflow 3 public DAG interface](https://airflow.apache.org/docs/apache-airflow/stable/public-airflow-interface.html)
- [Airflow best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
