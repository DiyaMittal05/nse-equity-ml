"""Load and validate JSON manifests from the registry."""

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class Manifest:
    name: str
    type: str  # "tool" or "agent"
    universe: str  # "psu", "nbfc", or "both"
    description: str
    capabilities: list[str]
    entrypoint: str
    parameters: dict[str, Any]
    output_schema: str
    sub_manifests: list[str]
    requires_llm: bool
    llm_prompt_template: str | None = None


class ManifestLoader:
    def __init__(self, manifest_dir: str | Path | None = None):
        if manifest_dir is None:
            self.manifest_dir = Path(__file__).resolve().parent.parent.parent / "data" / "manifests"
        else:
            self.manifest_dir = Path(manifest_dir)
        self._manifests: dict[str, Manifest] = {}
        self._load_all()

    def _load_all(self) -> None:
        for path in self.manifest_dir.glob("*.json"):
            with open(path, "r") as f:
                raw = json.load(f)
            m = Manifest(
                name=raw["name"],
                type=raw["type"],
                universe=raw["universe"],
                description=raw["description"],
                capabilities=raw.get("capabilities", []),
                entrypoint=raw["entrypoint"],
                parameters=raw.get("parameters", {}),
                output_schema=raw.get("output_schema", ""),
                sub_manifests=raw.get("sub_manifests", []),
                requires_llm=raw.get("requires_llm", False),
                llm_prompt_template=raw.get("llm_prompt_template"),
            )
            self._manifests[m.name] = m

    def get(self, name: str) -> Manifest | None:
        return self._manifests.get(name)

    def list_all(self) -> list[Manifest]:
        return list(self._manifests.values())

    def by_universe(self, universe: str) -> list[Manifest]:
        """Return manifests matching a universe (including 'both')."""
        return [m for m in self._manifests.values() if m.universe == universe or m.universe == "both"]

    def by_capability(self, capability: str) -> list[Manifest]:
        return [m for m in self._manifests.values() if capability in m.capabilities]

    def __len__(self) -> int:
        return len(self._manifests)
