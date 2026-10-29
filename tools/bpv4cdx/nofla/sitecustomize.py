# Put this directory on PYTHONPATH to make transformers fall back to its torch linear-attention kernels.
# This machine has no C compiler, so fla's Triton kernels that are not already in the Triton cache cannot be built.
import transformers.utils.import_utils as _iu

_iu.is_flash_linear_attention_available = lambda: False
