from __future__ import annotations

import json
from pathlib import Path


def test_packaged_retry_separation_fixture() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "retry-separation-v1.fixture.json").read_text()
    )
    assert fixture["contract"] == "clavenar.retry-separation/v1"
    cases = {case["id"]: case for case in fixture["cases"]}
    assert cases["explicit-side-effect-free-decision"]["automaticTransportRetry"] is True
    assert cases["explicit-side-effect-free-decision"]["maximumEffectAttempts"] == 0
    assert cases["sdk-registered-executor"]["automaticTransportRetry"] is False
    assert cases["sdk-registered-executor"]["maximumEffectAttempts"] == 1
    assert fixture["invariants"]["executorFailuresNeverEnterTransportRetryLoop"] is True
