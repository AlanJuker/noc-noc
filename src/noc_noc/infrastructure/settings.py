import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("NOC_CACHE", "data/cache"))
