import sys
from pathlib import Path

# make `softsignal` importable when pytest is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
