# Source this in every shell on the JKU student servers:   source scripts/env.sh
# Everything large lives in the 200 GB studentwork quota, never in the 20 GB home directory.

export WORK="${WORK:-/system/user/studentwork/$USER}"

# Hugging Face models and datasets
export HF_HOME="$WORK/hf_cache"
# uv: binary, cache, managed Pythons (the .venv itself lives inside the repo, which is under $WORK)
export UV_INSTALL_DIR="$WORK/bin"
export UV_CACHE_DIR="$WORK/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/.uv-python"
export PATH="$WORK/bin:$PATH"
# Data and large outputs
export INSPLAD_DIR="$WORK/data/insplad"
export FEDORCH_ARTIFACTS="$WORK/artifacts"
# Torch / pip temp files can be large; keep them off the home quota
export TMPDIR="$WORK/tmp"
export TORCH_HOME="$WORK/torch_cache"

# Institute recommendation (keep CPU libraries single-threaded per process)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-SEQUENTIAL}"
# Without TF installed this is harmless; with TF it prevents a known transformers/laya deadlock
export USE_TF=0

mkdir -p "$WORK"/{bin,hf_cache,.uv-cache,.uv-python,data,artifacts,tmp,torch_cache}

# Pick your reserved GPU(s) from the spreadsheet BEFORE running anything, e.g.:
#   export CUDA_VISIBLE_DEVICES=2
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[env.sh] WARNING: CUDA_VISIBLE_DEVICES is not set. Reserve a GPU in the spreadsheet and export it." >&2
fi
