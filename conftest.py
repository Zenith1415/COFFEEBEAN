import sys
from pathlib import Path

# Add project root to sys.path so `src` is importable in tests
sys.path.insert(0, str(Path(__file__).parent))
