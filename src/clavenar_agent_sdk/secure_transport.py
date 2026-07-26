"""Reusable mutual-TLS transport profile."""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx

from clavenar_agent_sdk.errors import ClavenarConfigError


class TokenProvider(Protocol):
    """Acquire the current bearer token without persisting it in the profile."""

    def __call__(self) -> str | None: ...


@dataclass(frozen=True)
class ProxyPolicy:
    """Explicit proxy behavior; ambient variables are opt-in."""

    mode: Literal["direct", "environment", "explicit"] = "direct"
    url: str | None = None


@dataclass(frozen=True)
class SecureTransportProfile:
    """Complete reload-before-request TLS, token, timeout, and proxy profile."""

    ca_bundle_path: Path
    client_certificate_path: Path
    private_key_path: Path
    token_provider: TokenProvider | None = None
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 10.0
    proxy: ProxyPolicy = ProxyPolicy()

    def token(self) -> str | None:
        """Acquire and validate the current token for one request."""
        token = self.token_provider() if self.token_provider is not None else None
        if token is None:
            return None
        value = token.strip()
        if not value:
            raise ClavenarConfigError("secure transport token provider returned an empty token")
        return value

    def async_client(self) -> httpx.AsyncClient:
        """Build a fresh complete async snapshot from the current source files."""
        context, timeout, trust_env, proxy = self._client_components()
        return httpx.AsyncClient(
            verify=context,
            timeout=timeout,
            trust_env=trust_env,
            proxy=proxy,
        )

    def client(self) -> httpx.Client:
        """Build a fresh complete sync snapshot from the current source files."""
        context, timeout, trust_env, proxy = self._client_components()
        return httpx.Client(
            verify=context,
            timeout=timeout,
            trust_env=trust_env,
            proxy=proxy,
        )

    def _client_components(
        self,
    ) -> tuple[ssl.SSLContext, httpx.Timeout, bool, str | None]:
        if self.connect_timeout_s <= 0 or self.request_timeout_s <= 0:
            raise ClavenarConfigError("secure transport timeouts must be positive")
        ca_bundle = self._required(self.ca_bundle_path, "CA bundle")
        context = ssl.create_default_context(cafile=str(ca_bundle))
        cert = self._required(self.client_certificate_path, "client certificate")
        key = self._required(self.private_key_path, "private key")
        try:
            context.load_cert_chain(certfile=str(cert), keyfile=str(key))
        except (OSError, ssl.SSLError) as error:
            raise ClavenarConfigError(
                f"invalid secure transport client identity {cert}: {error}"
            ) from error

        timeout = httpx.Timeout(self.request_timeout_s, connect=self.connect_timeout_s)
        proxy_url: str | None = None
        if self.proxy.mode == "explicit":
            if not self.proxy.url or not self.proxy.url.startswith(("http://", "https://")):
                raise ClavenarConfigError("secure transport explicit proxy must use HTTP or HTTPS")
            proxy_url = self.proxy.url
        elif self.proxy.mode not in {"direct", "environment"}:
            raise ClavenarConfigError(f"unknown secure transport proxy mode: {self.proxy.mode}")
        return context, timeout, self.proxy.mode == "environment", proxy_url

    @staticmethod
    def _required(path: Path, label: str) -> Path:
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise ClavenarConfigError(f"secure transport {label} {path} is missing or empty")
        except OSError as error:
            raise ClavenarConfigError(
                f"cannot read secure transport {label} {path}: {error}"
            ) from error
        return path
