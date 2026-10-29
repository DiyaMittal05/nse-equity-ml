"""Load and query the local PSU+NBFC knowledge base."""

import json
from pathlib import Path
from src.config.settings import KB_PATH


class KBLoader:
    def __init__(self, kb_path: str | Path | None = None):
        self.kb_path = Path(kb_path) if kb_path else KB_PATH
        self._kb: dict | None = None
        self._alias_map: dict[str, str] = {}  # alias -> ticker
        self._company_map: dict[str, dict] = {}  # ticker -> company
        self._load()

    def _load(self) -> None:
        with open(self.kb_path, "r") as f:
            self._kb = json.load(f)
        for comp in self._kb.get("companies", []):
            ticker = comp["ticker"]
            self._company_map[ticker] = comp
            for alias in comp.get("aliases", []):
                self._alias_map[alias.lower()] = ticker
            # also map exact name
            self._alias_map[comp["name"].lower()] = ticker
            self._alias_map[ticker.lower()] = ticker

    def resolve(self, text: str) -> list[dict]:
        """Find all companies mentioned in text."""
        text_lower = text.lower()
        found = set()
        for alias, ticker in self._alias_map.items():
            if alias in text_lower:
                found.add(ticker)
        return [self._company_map[t] for t in found]

    def get(self, ticker: str) -> dict | None:
        return self._company_map.get(ticker)

    def by_universe(self, universe: str) -> list[dict]:
        return [c for c in self._kb.get("companies", []) if c.get("universe") == universe]

    def sector_beta(self, sub_sector: str) -> float:
        return self._kb.get("sector_betas", {}).get(sub_sector, 1.0)
