"""Quick smoke tests for Phase 1 components."""

import sys

from src.registry import ManifestLoader
from src.routing import UniverseRouter
from src.moderation import ContentFilter
from src.tools import ohlcv_run, indicator_run


def test_manifest_loader():
    loader = ManifestLoader()
    assert len(loader) == 8, f"Expected 8 manifests, got {len(loader)}"
    m = loader.get("ohlcv_psu")
    assert m is not None
    assert m.type == "tool"
    assert m.universe == "psu"
    psu_manifests = loader.by_universe("psu")
    assert len(psu_manifests) >= 2
    print(f"  [PASS] ManifestLoader: {len(loader)} manifests loaded")


def test_router():
    router = UniverseRouter()
    r = router.route("What is the impact on SBIN?", "RBI hikes repo rate")
    assert "psu" in r["universes"], f"Expected psu in {r}"
    assert r["macro"] is True
    assert "SBIN.NS" in r["tickers"]
    print(f"  [PASS] Router: {r}")

    r2 = router.route("Compare BAJFINANCE with CHOLAFIN")
    assert "nbfc" in r2["universes"]
    assert "BAJFINANCE.NS" in r2["tickers"]
    print(f"  [PASS] Router NBFC: {r2}")


def test_moderation():
    mod = ContentFilter()
    r = mod.check("What is the impact on SBIN?")
    assert r["status"] == "PASS"
    print(f"  [PASS] Moderation PASS: {r}")

    r2 = mod.check("Here is a recipe for biryani")
    assert r2["status"] == "REJECT"
    print(f"  [PASS] Moderation REJECT: {r2}")


def test_ohlcv_tool():
    df = ohlcv_run("SBIN.NS", "psu")
    assert len(df) > 0
    assert "close" in df.columns
    print(f"  [PASS] OHLCV Tool: {len(df)} rows for SBIN.NS")


def test_indicator_tool():
    rsi = indicator_run("SBIN.NS", "psu", "RSI")
    assert len(rsi) > 0
    print(f"  [PASS] Indicator Tool: RSI series length {len(rsi)}")


if __name__ == "__main__":
    print("Running Phase 1 component tests...")
    test_manifest_loader()
    test_router()
    test_moderation()
    test_ohlcv_tool()
    test_indicator_tool()
    print("\nAll Phase 1 tests passed.")


def test_router_resolves_company_names():
    router = UniverseRouter()
    r = router.route("Bajaj Finance reports record profit growth")
    assert r["universes"] == ["nbfc"], r
    assert "BAJFINANCE.NS" in r["tickers"]

    r2 = router.route("Punjab National Bank NPAs fall")
    assert "psu" in r2["universes"] and "PNB.NS" in r2["tickers"], r2
