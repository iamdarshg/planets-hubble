"""The network boundary for MAST JSON requests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class JsonTransport(Protocol):
    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST a JSON MAST request and return its decoded JSON object."""


class MastTransportError(RuntimeError):
    """Raised when the MAST transport cannot return a JSON object."""


class MastJsonTransport:
    """Dependency-light urllib adapter for the MAST invoke endpoint.

    The opener is injectable so request encoding and transport failures can be
    tested without contacting MAST. No authorization header is ever added.
    """

    def __init__(
        self,
        *,
        endpoint: str = "https://mast.stsci.edu/api/v0/invoke",
        opener: Callable[..., Any] = urlopen,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self._opener = opener
        self.timeout = timeout

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            url or self.endpoint,
            data=urlencode(
                {"request": json.dumps(dict(payload), separators=(",", ":"))}
            ).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
            decoded = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # urllib and JSON errors have one public boundary
            raise MastTransportError(f"MAST request failed: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise MastTransportError("MAST response was not a JSON object")
        return decoded
