"""Reusable mutual-TLS transport profile with explicit rotation lifecycle."""

from __future__ import annotations

import math
import ssl
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

import httpx

from clavenar_agent_sdk.errors import ClavenarConfigError

MAX_SECURE_TIMEOUT_S = 300.0


class TokenProvider(Protocol):
    """Acquire the current bearer token without persisting it in the profile."""

    def __call__(self) -> str | None: ...


@dataclass(frozen=True)
class ProxyPolicy:
    """Explicit proxy behavior; ambient variables are opt-in."""

    mode: Literal["direct", "environment", "explicit"] = "direct"
    url: str | None = None


@dataclass
class SecureTransportProfile:
    """Cached mTLS clients with atomic, explicit credential reload.

    TLS material is loaded when a sync or async client is first needed.
    Calls then reuse its connection pool. Use :meth:`reload` (async) or
    :meth:`reload_sync` (sync-only profiles) after rotating certificate
    files, and close the profile during application shutdown.
    """

    ca_bundle_path: Path
    client_certificate_path: Path
    private_key_path: Path
    token_provider: TokenProvider | None = None
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 10.0
    proxy: ProxyPolicy = ProxyPolicy()
    _async_client: httpx.AsyncClient | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _sync_client: httpx.Client | None = field(default=None, init=False, repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def token(self) -> str | None:
        """Acquire and validate the current token for one request."""
        with self._lock:
            self._ensure_open()
        token = self.token_provider() if self.token_provider is not None else None
        if token is None:
            return None
        value = token.strip()
        if not value:
            raise ClavenarConfigError("secure transport token provider returned an empty token")
        if "\r" in value or "\n" in value:
            raise ClavenarConfigError("secure transport token provider returned a multi-line token")
        return value

    def async_client(self) -> httpx.AsyncClient:
        """Return the cached async client, creating one complete TLS snapshot if needed."""
        with self._lock:
            self._ensure_open()
            if self._async_client is None:
                self._async_client = self._new_async_client()
            return self._async_client

    def client(self) -> httpx.Client:
        """Return the cached sync client, creating one complete TLS snapshot if needed."""
        with self._lock:
            self._ensure_open()
            if self._sync_client is None:
                self._sync_client = self._new_sync_client()
            return self._sync_client

    async def reload(self) -> None:
        """Atomically replace all active TLS clients, then close the old snapshots."""
        with self._lock:
            self._ensure_open()
            had_async = self._async_client is not None
            had_sync = self._sync_client is not None
            new_sync = self._new_sync_client() if had_sync else None
            try:
                new_async = self._new_async_client() if had_async else None
            except Exception:
                if new_sync is not None:
                    new_sync.close()
                raise
            old_async = self._async_client
            old_sync = self._sync_client
            self._async_client = new_async
            self._sync_client = new_sync
        if old_sync is not None:
            old_sync.close()
        if old_async is not None:
            await old_async.aclose()

    def reload_sync(self) -> None:
        """Reload a sync-only profile without creating an event loop implicitly."""
        with self._lock:
            self._ensure_open()
            if self._async_client is not None:
                raise ClavenarConfigError(
                    "secure transport has an async client; await reload() instead"
                )
            new_sync = self._new_sync_client() if self._sync_client is not None else None
            old_sync = self._sync_client
            self._sync_client = new_sync
        if old_sync is not None:
            old_sync.close()

    async def aclose(self) -> None:
        """Close sync and async pools. The profile cannot be reused afterwards."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            async_client = self._async_client
            sync_client = self._sync_client
            self._async_client = None
            self._sync_client = None
        if sync_client is not None:
            sync_client.close()
        if async_client is not None:
            await async_client.aclose()

    def close(self) -> None:
        """Close a sync-only profile. Async profiles must use :meth:`aclose`."""
        with self._lock:
            if self._closed:
                return
            if self._async_client is not None:
                raise ClavenarConfigError(
                    "secure transport has an async client; await aclose() instead"
                )
            self._closed = True
            sync_client = self._sync_client
            self._sync_client = None
        if sync_client is not None:
            sync_client.close()

    def _new_async_client(self) -> httpx.AsyncClient:
        context, timeout, trust_env, proxy = self._client_components()
        return httpx.AsyncClient(
            verify=context,
            timeout=timeout,
            trust_env=trust_env,
            proxy=proxy,
        )

    def _new_sync_client(self) -> httpx.Client:
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
        if (
            not math.isfinite(self.connect_timeout_s)
            or not math.isfinite(self.request_timeout_s)
            or not 0 < self.connect_timeout_s <= MAX_SECURE_TIMEOUT_S
            or not 0 < self.request_timeout_s <= MAX_SECURE_TIMEOUT_S
        ):
            raise ClavenarConfigError(
                "secure transport timeouts must be positive, finite, and no greater than "
                f"{MAX_SECURE_TIMEOUT_S}s"
            )
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
            proxy_url = self._validate_proxy_url(self.proxy.url)
        elif self.proxy.mode not in {"direct", "environment"}:
            raise ClavenarConfigError(f"unknown secure transport proxy mode: {self.proxy.mode}")
        elif self.proxy.url is not None:
            raise ClavenarConfigError(
                "secure transport proxy URL is valid only when proxy.mode is 'explicit'"
            )
        return context, timeout, self.proxy.mode == "environment", proxy_url

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClavenarConfigError("secure transport profile is closed")

    @staticmethod
    def _validate_proxy_url(value: str | None) -> str:
        if not value:
            raise ClavenarConfigError("secure transport explicit proxy requires a URL")
        try:
            parsed = urlparse(value)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as error:
            raise ClavenarConfigError(f"secure transport proxy URL is invalid: {error}") from error
        if parsed.scheme not in {"http", "https"} or hostname is None or port == 0:
            raise ClavenarConfigError("secure transport explicit proxy must use HTTP or HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise ClavenarConfigError("secure transport proxy URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ClavenarConfigError(
                "secure transport proxy URL must not contain a query or fragment"
            )
        return value

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
