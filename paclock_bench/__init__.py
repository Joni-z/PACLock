"""PACLock benchmark package."""
import os as _os

# AMD's MI210 single-card VM fails in the default hipBLASLt path. The SLURM
# environment opts into rocBLAS; all other runtimes retain their own defaults.
if _os.environ.get('PACLOCK_USE_ROCBLAS') == '1':
    if not _os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('PACLOCK_USE_ROCBLAS requires a compute allocation')
    import torch as _torch
    _torch.backends.cuda.preferred_blas_library('cublas')
