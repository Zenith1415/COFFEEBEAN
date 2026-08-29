# COFFEEBEAN

**SIH26052 — AI/ML-enabled Adaptive Noise Cancellation (ANC)**

## Purpose

COFFEEBEAN is an MLOps platform for developing, training, and deploying adaptive noise cancellation systems. The project provides a reproducible foundation for audio processing workflows, from data preprocessing to model deployment.

## SIH26052

This project participates in Smart India Hackathon 2026 (SIH26052), focusing on AI/ML-enabled adaptive noise cancellation technology.

## MLOps Platform vs Edge Responsibilities

- **MLOps Platform (Cloud)**: Provides the infrastructure for training, experimentation, and model versioning. Handles large-scale training runs, dataset management, and model artifact storage.
- **Edge Responsibility**: Deploy optimized ANC models to edge devices (embedded systems, smartphones, IoT devices) with real-time noise cancellation capabilities.

## Current Phase 1 Status

Phase 1 establishes the local MLOps foundation:
- Python 3.11 virtual environment with DVC for data versioning
- Project structure with standardized directories
- Configuration management via YAML
- Git and GitHub Actions workflow
- Reproducible Docker environment

## Current Technology Stack

- Python 3.11.15
- DVC 3.67.1 for data version control
- Git for source version control
- YAML configuration
- pytest for testing
- Docker for containerization (base image: python:3.11-slim)

## Future Phases

Phase 2 will include:
- MLflow experiment tracking
- MinIO model/object storage
- PyTorch framework integration
- Kubernetes deployment
- Edge model optimization and deployment
- TensorRT acceleration
- ONNX model conversion
- Cloud infrastructure setup