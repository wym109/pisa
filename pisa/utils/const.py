import numpy as np


__all__ = ['FTYPE', 'C_FTYPE', 'C_PRECISION_DEF']


FTYPE = np.float64
"""Global floating-point data type. C, CUDA, and Numba datatype definitions are
derived from this"""


if FTYPE == np.float32:
    C_FTYPE = 'float'
    C_PRECISION_DEF = 'SINGLE_PRECISION'
elif FTYPE == np.float64:
    C_FTYPE = 'double'
    C_PRECISION_DEF = 'DOUBLE_PRECISION'
else:
    raise ValueError('FTYPE must be one of `np.float32` or `np.float64`. Got'
                     ' %s instead.' %FTYPE)
