export FEDORCH_WORK="${FEDORCH_WORK:-/system/user/studentwork/$USER}"
export FEDORCH_DATA="$FEDORCH_WORK/data"
export INSPLAD_DIR="$FEDORCH_DATA/insplad"

export UV_INSTALL_DIR="$FEDORCH_WORK/bin"
export UV_CACHE_DIR="$FEDORCH_WORK/cache/uv"
export UV_PYTHON_INSTALL_DIR="$FEDORCH_WORK/uv-python"
export UV_LINK_MODE=copy
export PATH="$FEDORCH_WORK/bin:$PATH"

export HF_HOME="$FEDORCH_WORK/cache/huggingface"
export TORCH_HOME="$FEDORCH_WORK/cache/torch"
export TRITON_CACHE_DIR="$FEDORCH_WORK/cache/triton"
export XDG_CACHE_HOME="$FEDORCH_WORK/cache/xdg"
export PIP_CACHE_DIR="$FEDORCH_WORK/cache/pip"
if [ -d /dev/shm ]; then export TMPDIR="/dev/shm/$USER-fedorch"; else export TMPDIR="$FEDORCH_WORK/tmp"; fi
export USE_TF=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_THREADING_LAYER=SEQUENTIAL

mkdir -p "$FEDORCH_WORK"/{bin,cache,tmp,uv-python} "$XDG_CACHE_HOME" "$INSPLAD_DIR" "$TMPDIR"

[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || echo "Set CUDA_VISIBLE_DEVICES to your booked GPU."
echo "FEDORCH_WORK=$FEDORCH_WORK"
