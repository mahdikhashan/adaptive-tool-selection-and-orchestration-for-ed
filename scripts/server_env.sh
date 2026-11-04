# Source this on the JKU AI-lab student servers before any other command:
#     source scripts/server_env.sh
#
# Keeps everything big (code, venv, caches, models, data, outputs) in the 200 GB studentwork
# directory and nothing in the 20 GB home directory.

export FEDORCH_WORK="${FEDORCH_WORK:-/system/user/studentwork/$USER}"

# Python tooling
export UV_INSTALL_DIR="$FEDORCH_WORK/bin"                 # uv binary
export UV_CACHE_DIR="$FEDORCH_WORK/cache/uv"
export UV_PYTHON_INSTALL_DIR="$FEDORCH_WORK/uv-python"
export UV_LINK_MODE=copy                                 # cache and venv may sit on different mounts
export PATH="$FEDORCH_WORK/bin:$PATH"

# Model and framework caches
export HF_HOME="$FEDORCH_WORK/cache/huggingface"
export TORCH_HOME="$FEDORCH_WORK/cache/torch"
export TRITON_CACHE_DIR="$FEDORCH_WORK/cache/triton"
export XDG_CACHE_HOME="$FEDORCH_WORK/cache/xdg"
export PIP_CACHE_DIR="$FEDORCH_WORK/cache/pip"
export TMPDIR="$FEDORCH_WORK/tmp"
export USE_TF=0                                          # stops transformers probing TensorFlow (Laya can hang)

# Data locations used by the scripts
export FEDORCH_DATA="$FEDORCH_WORK/data"                 # external datasets, LTE traces, ElecVQA clone
export INSPLAD_DIR="$FEDORCH_DATA/insplad"               # holds supervised_fault_classification/ and unsupervised_anomaly_detection/

# Institute recommendation for MKL; set torch threads explicitly inside scripts when needed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_THREADING_LAYER=SEQUENTIAL

mkdir -p "$FEDORCH_WORK"/{bin,cache,tmp,uv-python} "$FEDORCH_DATA" "$INSPLAD_DIR"

# GPU: pick a free GPU after booking it in the institute spreadsheet, e.g.
#     export CUDA_VISIBLE_DEVICES=0      # 0 = A10 on student06 (check nvidia-smi first)
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[server_env] CUDA_VISIBLE_DEVICES is not set: pick a free, booked GPU (see nvidia-smi)."
fi
echo "[server_env] FEDORCH_WORK=$FEDORCH_WORK  HF_HOME=$HF_HOME  INSPLAD_DIR=$INSPLAD_DIR"
