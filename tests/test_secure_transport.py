"""Secure transport profile contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from clavenar_agent_sdk.errors import ClavenarConfigError
from clavenar_agent_sdk.secure_transport import SecureTransportProfile


def test_token_is_acquired_and_trimmed_for_every_request(tmp_path: Path) -> None:
    generation = 0

    def token() -> str:
        nonlocal generation
        generation += 1
        return f" token-{generation} "

    profile = SecureTransportProfile(
        ca_bundle_path=tmp_path / "ca",
        client_certificate_path=tmp_path / "cert",
        private_key_path=tmp_path / "key",
        token_provider=token,
    )
    assert profile.token() == "token-1"
    assert profile.token() == "token-2"


def test_empty_token_fails_closed(tmp_path: Path) -> None:
    profile = SecureTransportProfile(
        ca_bundle_path=tmp_path / "ca",
        client_certificate_path=tmp_path / "cert",
        private_key_path=tmp_path / "key",
        token_provider=lambda: " ",
    )
    with pytest.raises(ClavenarConfigError, match="empty token"):
        profile.token()


def test_zero_timeout_fails_before_credentials_are_read(tmp_path: Path) -> None:
    profile = SecureTransportProfile(
        ca_bundle_path=tmp_path / "ca",
        client_certificate_path=tmp_path / "cert",
        private_key_path=tmp_path / "key",
        connect_timeout_s=0,
    )
    with pytest.raises(ClavenarConfigError, match="timeouts must be positive"):
        profile.client()


def test_sync_client_is_reused_and_close_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    profile = SecureTransportProfile(
        ca_bundle_path=tmp_path / "ca",
        client_certificate_path=tmp_path / "cert",
        private_key_path=tmp_path / "key",
    )
    client = httpx.Client()
    monkeypatch.setattr(profile, "_new_sync_client", lambda: client)
    assert profile.client() is client
    assert profile.client() is client
    profile.close()
    with pytest.raises(ClavenarConfigError, match="closed"):
        profile.client()
