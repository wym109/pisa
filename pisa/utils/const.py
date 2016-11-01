import os

import numpy as np


__all__ = ['FTYPE', 'C_FTYPE', 'C_PRECISION_DEF']


# Default value for FTYPE
FTYPE = np.float32
"""Global floating-point data type. C, CUDA, and Numba datatype definitions are
derived from this"""


# Set FTYPE from environment variable PISA_FTYPE, if it is defined
if 'PISA_FTYPE' in os.environ:
    pisa_ftype = os.environ['PISA_FTYPE']
    if pisa_ftype.strip().lower() in ['float', 'float32', '32', 'single']:
        FTYPE = np.float32
    elif pisa_ftype.strip().lower() in ['double', 'float64', '64']:
        FTYPE = np.float64
    else:
        raise ValueError('Environment var PISA_FTYPE val %s is unrecognized.'
                         %pisa_ftype)

"""
Derive #define consts for dynamically-compiled C (and also C++ and CUDA) code
to use.

To use these in code, put in the C/C++/CUDA the following at the TOP of your
code:

  from pisa import C_FTYPE, C_PRECISION_DEF

  ...

  dynamic_source_code = '''
  #define fType %(C_FTYPE)s
  #define %(C_PRECISION_DEF)s

  ...

  ''' % dict(C_FTYPE=C_FTYPE, C_PRECISION_DEF=C_PRECISION_DEF)

"""
if FTYPE == np.float32:
    C_FTYPE = 'float'
    C_PRECISION_DEF = 'SINGLE_PRECISION'
elif FTYPE == np.float64:
    C_FTYPE = 'double'
    C_PRECISION_DEF = 'DOUBLE_PRECISION'
else:
    raise ValueError('FTYPE must be one of `np.float32` or `np.float64`. Got'
                     ' %s instead.' %FTYPE)
