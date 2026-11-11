# Vendored third-party code

Copies of code the inference pipeline depends on, taken from `legacy/` (the original repository).
Keep changes to a minimum and list them here.

| path | source | changes |
|---|---|---|
| `pycochleagram/` (`__init__`, `cochleagram`, `erbfilter`, `subband`, `utils`) | `legacy/pycochleagram/`, originally https://github.com/mcdermottLab/pycochleagram | `from pycochleagram import X` → `from . import X`, so no top-level `pycochleagram` module is installed. `utils.py`: removed the display probe on import (it ran `python -c '…'` in a subprocess and switched matplotlib to Agg); pyplot is imported directly. `demo.py`, `misc.py` and the tests are not copied. |
| `NetBuilder_valid_pad.py` | `legacy/NetBuilder_valid_pad.py` | none |
