"""
Compatibility module for numpy constants
This module provides backward compatibility for code expecting older numpy naming conventions
"""

import numpy

# In newer numpy versions, NaN is lowercase 'nan'
NaN = numpy.nan

# Make other constants available if needed
Inf = numpy.inf 