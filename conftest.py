"""Make `pytest tests/` work from a bare checkout, with no prior `pip install -e .`.

Without this, collection fails with `ModuleNotFoundError: No module named
'tenor_saxs'` in a fresh clone -- the editable install remains the
documented (and preferred) workflow; this is just a fallback for the case
where a reader runs the tests before installing anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
