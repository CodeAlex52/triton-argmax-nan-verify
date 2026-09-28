import os
os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import triton
import triton.language as tl
from triton.language import core as tlc


# Custom wrappers: NOT identity-equal to the standard combine functions, so
# the interpreter's np.nanargmax special case does not short-circuit and the
# generic reduction path executes the actual combine logic under test.
@triton.jit
def _wrap_max_combine(v1, i1, v2, i2):
    return tl.standard._argmax_combine(v1, i1, v2, i2, True)


@triton.jit
def _wrap_min_combine(v1, i1, v2, i2):
    return tl.standard._argmin_combine(v1, i1, v2, i2, True)


@triton.jit
def reduce_kernel(x_ptr, max_i_ptr, min_i_ptr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    _, max_i = tlc._reduce_with_indices(x, 0, _wrap_max_combine)
    _, min_i = tlc._reduce_with_indices(x, 0, _wrap_min_combine)
    tl.store(max_i_ptr, max_i.to(tl.int32))
    tl.store(min_i_ptr, min_i.to(tl.int32))


x = np.array([-13.0, -1.0, 17.0, np.nan], dtype=np.float32)
max_i = np.zeros(1, dtype=np.int32)
min_i = np.zeros(1, dtype=np.int32)
reduce_kernel[(1, )](x, max_i, min_i, N=4)
print("max_idx:", max_i[0], "min_idx:", min_i[0])
assert max_i[0] == 2, f"argmax index must be 2 (finite max 17.0), got {max_i[0]}"
assert min_i[0] == 0, f"argmin index must be 0 (finite min -13.0), got {min_i[0]}"
print("PASS: argmax/argmin indices follow the nan-ignore value semantics")
