"""Allow ``python -m blcnn ...``.

On Windows with Python 3.7, the console-script launcher that uv generates
(``blcnn.exe``) does not start, so ``python -m blcnn`` is the portable way
to run the CLI.
"""

import sys

from .cli import main

sys.exit(main())
