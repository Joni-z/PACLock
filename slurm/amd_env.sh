#!/bin/bash
# Source inside an AMD allocation. Torch 2.7.1 targets CPython 3.9;
# torch 2.10.0 targets CPython 3.12. Do not mix their extension modules.
case "${SLURM_JOB_PARTITION:?AMD SLURM allocation required}" in
  mi2101x) default_module=pytorch/2.7.1 ;;
  mi3501x) default_module=pytorch/2.10.0 ;;
  mi2104x|mi2508x) default_module=pytorch/2.7.1 ;;
  *) echo "Unsupported AMD partition: $SLURM_JOB_PARTITION" >&2; return 2 ;;
esac
selected_module=${PACLOCK_TORCH_MODULE:-$default_module}
if [[ "$SLURM_JOB_PARTITION" == mi2101x && "$selected_module" == pytorch/2.7.1 ]]; then
  # This VM lacks the module's hardcoded rocm/6.3.1 dependency. Use the same
  # shared torch wheel with its bundled libraries on the available ROCm host.
  module load rocm/6.4.1
  export PYTHONPATH="/share/sw/ai/pytorch/2.7.1${PYTHONPATH:+:$PYTHONPATH}"
  export PACLOCK_USE_ROCBLAS=1
  export ROCBLAS_USE_HIPBLASLT=0
  export DISABLE_ADDMM_CUDA_LT=1
  export ROCBLAS_TENSILE_LIBPATH=/share/sw/ai/pytorch/2.7.1/torch/lib/rocblas/library
  export HIPBLASLT_TENSILE_LIBPATH=/share/sw/ai/pytorch/2.7.1/torch/lib/hipblaslt/library
else
  module load "$selected_module"
fi
case "$SLURM_JOB_PARTITION" in
  mi3501x)
    # Separate CPython 3.12 dependencies; never mix the two torch module trees.
    deps=${PACLOCK_AMD_DEPS:-/work1/chenyuyou/yifanwang/Zhizhe/PACLock-runtime/py312-v1}
    test -d "$deps/einops" && test -d "$deps/sklearn" || {
      echo "Missing single-card dependencies at $deps; see slurm/amd-py312-requirements.txt" >&2; return 2;
    }
    export PYTHONPATH="$deps${PYTHONPATH:+:$PYTHONPATH}"
    ;;
esac
case "$selected_module" in
  pytorch/2.7.1) default_python=python3.9 ;;
  pytorch/2.10.0) default_python=python3.12 ;;
  *) echo "Set up a supported PyTorch module (2.7.1 or 2.10.0)" >&2; return 2 ;;
esac
PACLOCK_PYTHON=${PACLOCK_PYTHON:-$default_python}
command -v "$PACLOCK_PYTHON" >/dev/null || { echo "Missing $PACLOCK_PYTHON for $selected_module" >&2; return 2; }
export PACLOCK_PYTHON
export PACLOCK_TORCH_MODULE=$selected_module
