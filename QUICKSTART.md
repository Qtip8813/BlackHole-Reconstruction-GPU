# QRFT Black Hole Toolkit - Quick Start Guide

## Overview
The QRFT Black Hole Toolkit has been containerized with Docker. This setup provides:
- Multi-stage optimized builds (50%+ smaller runtime images)
- GPU support with CUDA/CuPy configuration via `.env`
- Hot-reload development with file watching
- Jupyter lab for interactive analysis (optional)
- Resource limits and health checks

## Files Created

```
Dockerfile           # Multi-stage build (builder → runtime)
docker-compose.yml   # Services, volumes, networking
.env                 # Environment variables (GPU, resources, etc.)
.dockerignore        # Build context optimization
```

## Quick Start

### 1. Build the Docker image
```bash
docker build -f Dockerfile --target runtime -t qrft-blackhole-toolkit:latest .
```
Expected time: 5-10 minutes on first build (subsequent builds cached)

### 2. Run with Docker Compose

#### Standard run (CPU mode, default):
```bash
docker compose up
```

#### Watch mode (hot-reload on file changes):
```bash
docker compose up --watch
```

#### Run a specific test:
```bash
docker compose run qrft-toolkit -m tests.test_synthetic
docker compose run qrft-toolkit -m tests.test_phase3
docker compose run qrft-toolkit -m tests.test_phase4
```

#### Interactive shell:
```bash
docker compose exec qrft-toolkit bash
```

### 3. GPU Configuration (Optional)

If you have NVIDIA GPU hardware:

Edit `.env`:
```bash
ENABLE_GPU=true
CUDA_VERSION=12.x          # or 11.x for older CUDA
NVIDIA_VISIBLE_DEVICES=all # or 0,1 for multi-GPU
```

Update docker-compose.yml to use nvidia runtime:
```yaml
runtime: nvidia
```

Then rebuild and run:
```bash
docker compose up --build
```

### 4. Jupyter Lab (Interactive Analysis)

Start Jupyter with the jupyter profile:
```bash
docker compose --profile jupyter up
```

Access at: `http://localhost:8888`
Token: `qrft_token_2025` (configurable in `.env` as `JUPYTER_TOKEN`)

## Environment Variables (.env)

Key variables for customization:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_GPU` | false | Enable GPU/CUDA support |
| `CUDA_VERSION` | 12.x | CUDA version for CuPy |
| `COMPOSE_CPU_LIMIT` | 4 | Max CPU cores |
| `COMPOSE_MEM_LIMIT` | 8gb | Max memory |
| `OPENBLAS_NUM_THREADS` | 4 | Linear algebra threads |
| `JUPYTER_PORT` | 8888 | Jupyter port |
| `JUPYTER_TOKEN` | qrft_token_2025 | Jupyter auth token |

## Common Commands

```bash
# Rebuild after dependency changes
docker compose up --build

# View logs
docker compose logs -f qrft-toolkit

# Stop containers
docker compose down

# Remove volumes (cleanup)
docker compose down -v

# Check resource usage
docker stats

# Run with custom command
docker compose run qrft-toolkit python script.py

# Access shell
docker compose exec qrft-toolkit python
```

## Volume Mounts

- `./qrft-blackhole-toolkit/` → `/app` (source code, hot-reload)
- `output_data` → `/app/output` (persistent results)
- `eht_data` → `/app/eht_data` (persistent EHT datasets)
- `notebooks` → `/app/notebooks` (Jupyter notebooks, with jupyter profile)

## Health Check

The container includes a health check that validates core imports:
```dockerfile
HEALTHCHECK ... CMD python -c "from core import Q4StokesEncoder; Q4StokesEncoder()"
```

View health status:
```bash
docker compose ps
```

## Troubleshooting

### Build timeout
Increase Docker build timeout or run with increased resources:
```bash
docker build --build-arg BUILDKIT_PROGRESS=plain . -t qrft-blackhole-toolkit:latest
```

### Out of memory
Adjust in `.env`:
```bash
COMPOSE_MEM_LIMIT=16gb
COMPOSE_MEM_RESERVATION=8gb
```

### GPU not detected
Ensure NVIDIA Docker runtime is installed:
```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.0-base nvidia-smi
```

If that fails, install nvidia-docker or use CPU mode (`ENABLE_GPU=false`).

### Permission issues
Add your user to docker group:
```bash
sudo usermod -aG docker $USER
```

## Next Steps

1. Build the image: `docker build -f Dockerfile --target runtime -t qrft-blackhole-toolkit:latest .`
2. Start compose: `docker compose up`
3. Run tests: `docker compose run qrft-toolkit -m tests.test_synthetic`
4. Access Jupyter: `docker compose --profile jupyter up` → `http://localhost:8888`

## Performance Notes

- **First build**: 5-10 minutes (downloads base image, compiles dependencies)
- **Subsequent builds**: <1 minute (cached layers)
- **Runtime CPU**: ~2-4 cores typical for pipeline
- **GPU**: ~6GB VRAM for full model (configurable via CuPy memory pool)

---

For issues, check logs: `docker compose logs -f qrft-toolkit`

∞ 0425 - QRFT Black Hole Toolkit
