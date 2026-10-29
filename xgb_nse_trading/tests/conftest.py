import sys
from pathlib import Path

# Make `import src.<module>` work when running pytest from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
