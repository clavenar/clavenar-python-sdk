"""Environment-driven real-mTLS and rotation acceptance."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from clavenar_agent_sdk.options import ClavenarOptions
from clavenar_agent_sdk.secure_transport import SecureTransportProfile
from clavenar_agent_sdk.transport import NormalizedToolCall, inspect_tool_use


@pytest.mark.skipif(
    "CLAVENAR_SECURE_TRANSPORT_ENDPOINT" not in os.environ,
    reason="secure transport live endpoint not configured",
)
async def test_real_mtls_certificate_and_token_rotation() -> None:
    cert = Path(_required("CLAVENAR_SECURE_TRANSPORT_CLIENT_CERT"))
    key = Path(_required("CLAVENAR_SECURE_TRANSPORT_CLIENT_KEY"))
    generation = 0

    def token() -> str:
        nonlocal generation
        generation += 1
        return f"matrix-token-{generation}"

    profile = SecureTransportProfile(
        ca_bundle_path=Path(_required("CLAVENAR_SECURE_TRANSPORT_CA")),
        client_certificate_path=cert,
        private_key_path=key,
        token_provider=token,
    )
    options = ClavenarOptions(
        endpoint=_required("CLAVENAR_SECURE_TRANSPORT_ENDPOINT"),
        transport_profile=profile,
    )
    call = NormalizedToolCall(id="matrix", name="matrix_probe", input={})
    assert (await inspect_tool_use(call, options)).kind == "allow"

    shutil.copyfile(_required("CLAVENAR_SECURE_TRANSPORT_NEXT_CERT"), cert)
    shutil.copyfile(_required("CLAVENAR_SECURE_TRANSPORT_NEXT_KEY"), key)
    assert (await inspect_tool_use(call, options)).kind == "allow"
    assert generation == 2


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AssertionError(f"{name} is required")
    return value
