"""Method subpackages self-register on import.

Adding a new method: create a subpackage under this directory, apply the
`@register("...")` decorator to its DiffMethod subclass, and add one
import line here. No other file needs to change.
"""

from scripts.diffing.methods import activation_diff  # noqa: F401
from scripts.diffing.methods import pca  # noqa: F401
