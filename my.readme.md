# My nanochat

## AMD GPU Setup (ROCm 7.2 Nightly)

### Background & Requirements
* Hardware: AMD GPU (RDNA architecture, e.g. RX 9070 / 9700).
* ROCm 6.x had known compatibility issues with newer AMD GPUs, so PyTorch ROCm 7.2 nightly builds (`2.15.0.dev*`) were used.

### Configuration Changes
1. **`pyproject.toml`**:
   * Added `pytorch-rocm` index targeting `https://download.pytorch.org/whl/nightly/rocm7.2`.
   * Added an `[project.optional-dependencies]` entry for `amd`:
     ```toml
     amd = [
         "torch==2.15.0.dev20260815+rocm7.2",
     ]
     ```
   * Added `tool.uv.sources` mapping for `amd` extra to `pytorch-rocm`.
   * Added `tool.setuptools.packages.find` to prevent flat-layout package discovery errors during editable installs.

### Installation Steps
1. Create virtual environment:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

2. Install ROCm 7.2 PyTorch nightly wheels:
   ```bash
   uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2
   ```

3. Install project dependencies in editable mode:
   ```bash
   uv pip install -e .
   ```

### Verification
Run the following Python check to verify PyTorch ROCm backend and GPU detection:
```bash
python -c "import torch; import nanochat; print('Torch:', torch.__version__, '| ROCm HIP:', getattr(torch.version, 'hip', None), '| GPU available:', torch.cuda.is_available())"
```
**Expected Output:**
```text
Torch: 2.15.0.dev20260815+rocm7.2 | ROCm HIP: 7.2.53211 | GPU available: True
```

### Running Tests
```bash
uv pip install pytest
PYTHONPATH=. pytest -m "not slow"
```
*Result: 48 passed, 10 skipped.*

---

## Training Pipeline & Quick Check

### 1. Download Dataset Shards
Download the pretraining shards (e.g. first 8 shards for tokenizer training, or more for full pretraining):
```bash
uv run python -m nanochat.dataset -n 8
```

### 2. Train Tokenizer
Train the BPE tokenizer on downloaded text shards:
```bash
uv run python -m scripts.tok_train
uv run python -m scripts.tok_eval
```

### 3. Quick Pretraining Check (Smoke Test on AMD GPU)
Run a quick 5-step test to ensure training forward/backward passes and benchmark evaluations execute cleanly:
```bash
uv run python -m scripts.base_train \
    --depth=4 \
    --device-batch-size=8 \
    --num-iterations=5 \
    --window-pattern=L \
    --run=dummy
```
* Note: `--window-pattern=L` is recommended when Flash Attention 3 is not used to maximize SDPA performance.
* Benchmark check throughput observed: **~304,000 tokens/sec** on AMD Radeon AI PRO R9700.

### 4. Running Full Pretraining (Single GPU)
For a 12-layer model (GPT-1 size):
```bash
OMP_NUM_THREADS=1 uv run python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=16 \
    --window-pattern=L \
    --run="d12"
```

### Reproducibility
The installed environment dependencies are frozen in [`requirements-rocm.lock`](./requirements-rocm.lock). To replicate:
```bash
uv pip install -r requirements-rocm.lock --extra-index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

---

## Random observations

* Initial repo size: 3.4M
