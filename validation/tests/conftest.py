"""Put validation/ on sys.path so the tests can import the scripts the way the scripts import
each other (flat: `import common`, `import retrieval_common`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
