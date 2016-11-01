from pint import UnitRegistry
from ._version import get_versions
from pisa.utils.const import FTYPE, C_FTYPE, C_PRECISION_DEF


__all__ = ['ureg', 'Q_', 'FTYPE', 'C_FTYPE', 'C_PRECISION_DEF', '__version__']


ureg = UnitRegistry()
Q_ = ureg.Quantity

__version__ = get_versions()['version']
del get_versions
