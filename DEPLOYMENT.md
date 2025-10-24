# Deployment Guide for SSM-MetaRL-Unified

This guide covers various deployment options for the SSM-MetaRL-Unified framework.

---

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [PyPI Package Distribution](#pypi-package-distribution)
3. [GitHub Container Registry](#github-container-registry)
4. [Local Installation](#local-installation)
5. [Cloud Deployment](#cloud-deployment)

---

## Docker Deployment

### Building the Docker Image

```bash
# Build the image
docker build -t ssm-metarl-unified:latest .

# Verify the build
docker images | grep ssm-metarl-unified
```

### Running with Docker

**Basic Usage:**
```bash
# Run with default help command
docker run --rm ssm-metarl-unified:latest

# Run standard adaptation mode
docker run --rm ssm-metarl-unified:latest \
    python main.py --adaptation_mode standard --num_epochs 10

# Run hybrid adaptation mode
docker run --rm ssm-metarl-unified:latest \
    python main.py --adaptation_mode hybrid --buffer_size 5000 --num_epochs 10
```

**With Volume Mounts (for saving results):**
```bash
# Create results directory
mkdir -p results checkpoints

# Run with volume mounts
docker run --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/checkpoints:/app/checkpoints \
    ssm-metarl-unified:latest \
    python main.py --adaptation_mode hybrid --num_epochs 20
```

**Run Integration Tests:**
```bash
docker run --rm ssm-metarl-unified:latest python test_integration.py
```

**Run Benchmarks:**
```bash
docker run --rm \
    -v $(pwd)/results:/app/results \
    ssm-metarl-unified:latest \
    python experiments/quick_benchmark.py
```

### Using Docker Compose

Docker Compose provides a convenient way to manage multiple services.

**Start the main service:**
```bash
docker-compose up ssm-metarl-unified
```

**Run tests:**
```bash
docker-compose run test
```

**Run benchmarks:**
```bash
docker-compose run benchmark
```

**Run custom command:**
```bash
docker-compose run ssm-metarl-unified python main.py --adaptation_mode hybrid --buffer_size 10000
```

**Clean up:**
```bash
docker-compose down
```

### Docker Image Optimization

The Dockerfile uses multi-stage builds to minimize image size:

1. **Builder Stage**: Builds the Python wheel package
2. **Runtime Stage**: Uses slim Python image with only the wheel installed

**Image Size Comparison:**
- Without multi-stage: ~1.5 GB
- With multi-stage: ~500 MB

---

## PyPI Package Distribution

### Building the Package

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# - dist/ssm_metarl_unified-1.0.0-py3-none-any.whl
# - dist/ssm-metarl-unified-1.0.0.tar.gz
```

### Testing the Package Locally

```bash
# Install from local wheel
pip install dist/ssm_metarl_unified-1.0.0-py3-none-any.whl

# Test import
python -c "from core.ssm import StateSpaceModel; print('Success!')"

# Uninstall
pip uninstall ssm-metarl-unified
```

### Publishing to PyPI (Optional)

**Note**: This requires PyPI account credentials.

```bash
# Upload to TestPyPI first (recommended)
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ ssm-metarl-unified

# If everything works, upload to real PyPI
python -m twine upload dist/*
```

### Installing from PyPI (After Publishing)

```bash
# Basic installation
pip install ssm-metarl-unified

# With development dependencies
pip install ssm-metarl-unified[dev]

# With MuJoCo support
pip install ssm-metarl-unified[mujoco]

# With all optional dependencies
pip install ssm-metarl-unified[all]
```

---

## GitHub Container Registry

### Building and Pushing to GHCR

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build with GHCR tag
docker build -t ghcr.io/sunghunkwag/ssm-metarl-unified:latest .

# Tag with version
docker tag ghcr.io/sunghunkwag/ssm-metarl-unified:latest \
    ghcr.io/sunghunkwag/ssm-metarl-unified:1.0.0

# Push to GHCR
docker push ghcr.io/sunghunkwag/ssm-metarl-unified:latest
docker push ghcr.io/sunghunkwag/ssm-metarl-unified:1.0.0
```

### Pulling from GHCR

```bash
# Pull latest version
docker pull ghcr.io/sunghunkwag/ssm-metarl-unified:latest

# Pull specific version
docker pull ghcr.io/sunghunkwag/ssm-metarl-unified:1.0.0

# Run from GHCR
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-unified:latest \
    python main.py --adaptation_mode hybrid
```

### GitHub Actions for Automated Builds

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Docker Build and Publish

on:
  push:
    branches: [ master ]
    tags: [ 'v*.*.*' ]
  pull_request:
    branches: [ master ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

---

## Local Installation

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/sunghunkwag/SSM-MetaRL-Unified.git
cd SSM-MetaRL-Unified

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with all dependencies
pip install -e .[all]
```

### From GitHub Release

```bash
# Install directly from GitHub
pip install git+https://github.com/sunghunkwag/SSM-MetaRL-Unified.git

# Install specific version/tag
pip install git+https://github.com/sunghunkwag/SSM-MetaRL-Unified.git@v1.0.0
```

---

## Cloud Deployment

### AWS EC2

**1. Launch EC2 Instance:**
- Choose Ubuntu 22.04 LTS
- Instance type: t3.medium or larger
- Configure security group for SSH access

**2. Connect and Setup:**
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Logout and login again
exit
ssh -i your-key.pem ubuntu@your-instance-ip

# Pull and run
docker pull ghcr.io/sunghunkwag/ssm-metarl-unified:latest
docker run --rm ghcr.io/sunghunkwag/ssm-metarl-unified:latest \
    python main.py --adaptation_mode hybrid --num_epochs 50
```

### Google Cloud Platform (GCP)

**Using Google Compute Engine:**
```bash
# Create instance with Docker
gcloud compute instances create ssm-metarl-vm \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-4 \
    --boot-disk-size=50GB

# SSH into instance
gcloud compute ssh ssm-metarl-vm

# Install Docker and run (same as AWS)
```

**Using Google Kubernetes Engine (GKE):**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ssm-metarl-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ssm-metarl
  template:
    metadata:
      labels:
        app: ssm-metarl
    spec:
      containers:
      - name: ssm-metarl
        image: ghcr.io/sunghunkwag/ssm-metarl-unified:latest
        command: ["python", "main.py"]
        args: ["--adaptation_mode", "hybrid", "--num_epochs", "100"]
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Azure

**Using Azure Container Instances:**
```bash
# Create resource group
az group create --name ssm-metarl-rg --location eastus

# Create container instance
az container create \
    --resource-group ssm-metarl-rg \
    --name ssm-metarl-container \
    --image ghcr.io/sunghunkwag/ssm-metarl-unified:latest \
    --cpu 4 --memory 8 \
    --command-line "python main.py --adaptation_mode hybrid --num_epochs 100"

# View logs
az container logs --resource-group ssm-metarl-rg --name ssm-metarl-container
```

---

## Performance Optimization

### CPU Optimization

```bash
# Set number of threads
docker run --rm \
    -e OMP_NUM_THREADS=4 \
    ssm-metarl-unified:latest \
    python main.py --adaptation_mode hybrid
```

### Memory Optimization

```bash
# Limit memory usage
docker run --rm \
    --memory=4g \
    --memory-swap=4g \
    ssm-metarl-unified:latest \
    python main.py --adaptation_mode hybrid --buffer_size 5000
```

### GPU Support (Future)

For GPU support, use NVIDIA Docker runtime:

```bash
# Build with CUDA support (requires CUDA base image)
docker build -f Dockerfile.gpu -t ssm-metarl-unified:gpu .

# Run with GPU
docker run --rm --gpus all \
    ssm-metarl-unified:gpu \
    python main.py --adaptation_mode hybrid --device cuda
```

---

## Monitoring and Logging

### Docker Logs

```bash
# View logs
docker logs <container-id>

# Follow logs
docker logs -f <container-id>

# Save logs to file
docker logs <container-id> > training.log 2>&1
```

### Health Checks

The Dockerfile includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; import gymnasium; print('OK')" || exit 1
```

Check health status:
```bash
docker ps
# Look for "healthy" status
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory:**
```bash
# Reduce buffer size
python main.py --adaptation_mode hybrid --buffer_size 1000
```

**2. Docker Build Fails:**
```bash
# Clear Docker cache
docker builder prune -a

# Rebuild without cache
docker build --no-cache -t ssm-metarl-unified:latest .
```

**3. Import Errors:**
```bash
# Verify package installation
docker run --rm ssm-metarl-unified:latest \
    python -c "from core.ssm import StateSpaceModel; print('OK')"
```

---

## Security Considerations

### Docker Security

1. **Don't run as root** (already configured in Dockerfile)
2. **Use specific version tags** instead of `latest` in production
3. **Scan images for vulnerabilities:**
   ```bash
   docker scan ssm-metarl-unified:latest
   ```

### Secrets Management

For production deployments with sensitive data:

```bash
# Use Docker secrets
echo "my_secret_key" | docker secret create api_key -

# Use in container
docker service create \
    --secret api_key \
    ssm-metarl-unified:latest
```

---

## Continuous Integration/Deployment

### GitHub Actions Workflow

Create `.github/workflows/ci-cd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e .
      - name: Run tests
        run: python test_integration.py

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        # ... (see GitHub Actions section above)
```

---

## Summary

This deployment guide covers:

✅ **Docker**: Containerized deployment with multi-stage builds  
✅ **PyPI**: Python package distribution  
✅ **GHCR**: GitHub Container Registry integration  
✅ **Cloud**: AWS, GCP, Azure deployment options  
✅ **CI/CD**: Automated testing and deployment  

Choose the deployment method that best fits your use case:

- **Development**: Local installation from source
- **Production**: Docker containers with orchestration
- **Distribution**: PyPI package for easy installation
- **Research**: Google Colab for interactive exploration

For questions or issues, visit: https://github.com/sunghunkwag/SSM-MetaRL-Unified/issues

