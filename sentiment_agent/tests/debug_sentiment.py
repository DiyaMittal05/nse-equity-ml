import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.sentiment_model import TrainedSentiment

m = TrainedSentiment()
headlines = [
    'Bajaj Finance acquires new lending platform to expand rural reach',
    'SBI reports record profit growth of 15% in Q3',
    'RBI hikes repo rate by 25 bps',
    'PNB NPAs jump to 12%, highest in 8 quarters',
]
for h in headlines:
    d, c = m.predict(h)
    print(f'{h!r} -> direction={d}, confidence={c:.3f}')
