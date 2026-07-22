from __future__ import annotations

import json
from pathlib import Path


def test_packaged_client_migration_fixture() -> None:
    root = Path(__file__).parents[1] / "fixtures"
    fixture = json.loads((root / "client-migration-v1.fixture.json").read_text())
    schema = json.loads((root / "client-migration-v1.schema.json").read_text())
    assert fixture["contract"] == "clavenar.client-migration/v1"
    assert fixture["minimumSafeVersions"]["python"] == "1.4.0"
    assert fixture["legacyRejection"]["httpStatus"] == 426
    assert fixture["legacyRejection"]["executable"] is False
    assert fixture["legacyRejection"]["toolEffectCount"] == 0
    assert fixture["invariants"]["legacyInspectionCannotExecute"] is True
    assert schema["properties"]["contract"]["const"] == fixture["contract"]
