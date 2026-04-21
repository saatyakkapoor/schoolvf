"""Lightweight RTSP endpoint check: TCP connect to host:port only (no full RTSP handshake)."""

from __future__ import annotations

import socket
from urllib.parse import urlparse


def probe_rtsp_tcp(rtsp_url: str, *, timeout: float = 3.0) -> bool:
    """
    Return True if we can open a TCP connection to the RTSP host:port.

    This does not validate credentials or that the stream works — same as checking
    the camera is reachable from this machine. VLC can still work while this
    fails if the API runs in Docker without LAN access to the camera IP.
    """
    parsed = urlparse(rtsp_url.strip())
    if parsed.scheme not in ("rtsp", "rtsps"):
        return False
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or 554
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
