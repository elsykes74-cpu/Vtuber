#!/usr/bin/env python3
"""Launcher for duckduckgo-mcp-server with httpx 0.28+ compatibility patch.

duckduckgo-mcp-server uses httpx.TimeoutError which was removed in httpx 0.28.
This script patches httpx before importing the server so it uses TimeoutException.
"""

import httpx

# Patch for duckduckgo_mcp_server compatibility with httpx 0.28+
if not hasattr(httpx, "TimeoutError"):
    httpx.TimeoutError = httpx.TimeoutException

from duckduckgo_mcp_server.server import main

if __name__ == "__main__":
    main()
