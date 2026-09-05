# COFFEEBEAN — KubeEdge Fleet Architecture
# Phase 11: Future Architecture Document (NOT implemented yet)

## Overview

This document describes the target fleet architecture for COFFEEBEAN when the project
scales to multiple edge devices managed via KubeEdge.

> ⚠️ This is a planning document. KubeEdge deployment is a future phase.
> Current implementation uses standalone ONNX Runtime on each device.

---

## Architecture Diagram

```
                      ┌──────────────────────────────────────┐
                      │          CLOUD CONTROL PLANE         │
                      │                                      │
                      │  ┌─────────────┐  ┌──────────────┐  │
                      │  │  Kubernetes  │  │   DagsHub    │  │
                      │  │   Master     │  │  (MLflow +   │  │
                      │  │  (KubeEdge)  │  │    DVC)      │  │
                      │  └──────┬───────┘  └──────────────┘  │
                      └─────────┼────────────────────────────┘
                                │ EdgeCore ↔ CloudCore
                    ┌───────────┼────────────┐
                    ▼           ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │  Edge    │ │  Edge    │ │  Edge    │
            │ Device 1 │ │ Device 2 │ │ Device N │
            │(RPi / Jet│ │          │ │          │
            │son Nano) │ │          │ │          │
            │          │ │          │ │          │
            │ EdgeAgent│ │ EdgeAgent│ │ EdgeAgent│
            │ ONNX RT  │ │ ONNX RT  │ │ ONNX RT  │
            │ Monitoring│ │Monitoring│ │Monitoring│
            └──────────┘ └──────────┘ └──────────┘
```

---

## Components

### Cloud Control Plane
- **KubeEdge CloudCore** — manages edge node registration, model distribution
- **DagsHub** — MLflow tracking + DVC dataset versioning
- **MLflow Model Registry** — versioned model artifacts, staging/production stages
- **Airflow** — orchestrates training pipeline (runs on cloud)

### Edge Devices
- **KubeEdge EdgeCore** — runs on device, syncs with cloud, manages local pods
- **ANC Inference Pod** — ONNX Runtime container, no internet required
- **Monitoring Pod** — collects latency/CPU/RAM, buffers logs for cloud sync

---

## Model Update Flow

```
1. New model trained on Colab → logged to DagsHub MLflow
2. Quality Gate passes → model registered (Staging)
3. Team promotes to Production → MLflow registry updated
4. KubeEdge CloudCore detects new Production model
5. CloudCore sends model update command to all registered EdgeCores
6. Each EdgeCore downloads new ONNX model
7. EdgeCore restarts ANC Inference Pod with new model
8. Device reports back: "model_version=v5, status=running"
9. CloudCore aggregates health from all devices
```

---

## Prerequisites Before Implementation

| Item | Status |
|------|--------|
| KubeEdge installed on cloud VM | ⬜ Not done |
| Kubernetes cluster (>= 1 master + 1 worker) | ⬜ Not done |
| EdgeCore installed on Raspberry Pi / Jetson | ⬜ Not done |
| Device registered with CloudCore | ⬜ Not done |
| Helm chart for COFFEEBEAN ANC pod | ⬜ Not done |

---

## Cost Estimate (Phase 11 cloud infra)
- Kubernetes master node: ~$20/month (2 vCPU, 4GB RAM VM)
- DagsHub: Free tier (1GB storage)
- Edge devices: one-time hardware cost (RPi 4: ~$75)

---

## Next Steps to Implement Phase 11
1. Provision a small VM with k3s (lightweight Kubernetes)
2. Install KubeEdge CloudCore on VM
3. Install KubeEdge EdgeCore on Raspberry Pi
4. Create Helm chart for ANC inference pod
5. Add model version watcher to Airflow DAG
6. Implement model distribution pipeline
