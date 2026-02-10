"""Local HTTP server for the benchmark dashboard."""

from __future__ import annotations

import json
import threading
import urllib.parse
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from macsmart.benchmark.energy_compare import load_energy_results
from macsmart.benchmark.report import load_results

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Module-level server reference for stop_server()
_server: HTTPServer | None = None
_server_thread: threading.Thread | None = None


def results_to_json_api(results_dir: Path | None = None) -> list[dict]:
    """Serialize benchmark results for the JSON API.

    Args:
        results_dir: Directory to load results from.

    Returns:
        List of result dicts ready for JSON serialization.
    """
    from dataclasses import asdict

    results = load_results(results_dir=results_dir)
    return [asdict(r) for r in results]


def energy_results_to_json_api(results_dir: Path | None = None) -> list[dict]:
    """Serialize energy benchmark results for the JSON API.

    Args:
        results_dir: Directory to load results from.

    Returns:
        List of energy result dicts.
    """
    from dataclasses import asdict

    results = load_energy_results(results_dir=results_dir)
    out = []
    for r in results:
        out.append({
            "benchmark": asdict(r.benchmark),
            "energy": asdict(r.energy),
            "power_source": r.power_source,
        })
    return out


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving the dashboard SPA and API endpoints."""

    def __init__(self, *args, results_dir: Path | None = None, **kwargs) -> None:
        self.results_dir = results_dir
        super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/results":
            self._send_json(results_to_json_api(self.results_dir))
        elif path == "/api/energy-results":
            self._send_json(energy_results_to_json_api(self.results_dir))
        elif path == "/api/compare":
            self._handle_compare(parsed.query)
        else:
            super().do_GET()

    def _send_json(self, data: object) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_compare(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        idx_a = params.get("a", [None])[0]
        idx_b = params.get("b", [None])[0]

        if idx_a is None or idx_b is None:
            self._send_json({"error": "Provide ?a=<index>&b=<index>"})
            return

        try:
            a = int(idx_a)
            b = int(idx_b)
        except ValueError:
            self._send_json({"error": "Indices must be integers"})
            return

        from dataclasses import asdict

        from macsmart.benchmark.runner import compare_results

        results = load_results(results_dir=self.results_dir)
        if a >= len(results) or b >= len(results):
            self._send_json({"error": "Index out of range"})
            return

        comparison = compare_results(results[a], results[b])
        self._send_json(comparison)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default request logging."""


def start_server(
    port: int = 8377,
    results_dir: Path | None = None,
    open_browser: bool = True,
) -> HTTPServer:
    """Start the dashboard HTTP server.

    Args:
        port: Port to listen on.
        results_dir: Directory containing benchmark results.
        open_browser: Whether to auto-open the browser.

    Returns:
        The running HTTPServer instance.
    """
    global _server, _server_thread

    handler = partial(DashboardHandler, results_dir=results_dir)
    server = HTTPServer(("127.0.0.1", port), handler)
    _server = server

    _server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    _server_thread.start()

    if open_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")

    return server


def stop_server() -> None:
    """Stop the running dashboard server."""
    global _server, _server_thread
    if _server is not None:
        _server.shutdown()
        _server = None
    if _server_thread is not None:
        _server_thread.join(timeout=5.0)
        _server_thread = None
