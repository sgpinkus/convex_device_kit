from typing import Any

from numpy import number
from numpy.typing import NDArray

Number = int | float | NDArray[number]
Constraint = dict[str, Any] | Any  # {Constraint, dict}
