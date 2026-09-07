"""Local dashboard.

Built on ``http.server`` so the project keeps its zero-dependency property. It is
a development tool for inspecting a run, not a production service: it binds to
localhost, serves one page, and runs one batch per request.

The design decision worth noting is that a run completes **before** anything is
sent to the browser, and the page then replays the recorded timeline client
side. Streaming would have been more impressive to build and worse to use: the
whole point of a virtual clock is that three weeks of recovery simulate in a few
seconds, so there is nothing to stream. Replaying a finished run instead gives a
scrubbable timeline, which is what you actually want when explaining a decision,
and it keeps the server single threaded and free of races.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..clock import parse_iso
from ..eval.harness import CONFIGS
from ..money import fmt
from ..world.generator import generate_world

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# One run at a time. The runtime mutates world state, so two concurrent runs
# sharing a process would be fine (they build separate worlds) but would make
# timing figures meaningless and could exhaust memory on a large batch.
_RUN_LOCK = threading.Lock()

MAX_EVENTS = 4000


def build_run(agent_key: str, seed: int, items: int, horizon: int) -> dict[str, Any]:
    """Run one agent and return everything the page needs, in one pass.

    The runtime is constructed here rather than going through ``run_agent``
    because the dashboard needs the ledger as well as the metrics, and
    ``run_agent`` returns only metrics. An earlier version called ``run_agent``
    and then re-ran the whole batch to recover the ledger, which doubled every
    request for no reason.
    """
    from ..execute.razorpay_client import RazorpayClient
    from ..llm.gateway import LLMGateway
    from ..policy.engine import PolicyEngine
    from ..policy.rules import DEFAULT_RULES
    from ..runtime import Runtime

    if agent_key not in CONFIGS:
        raise ValueError(f"unknown agent {agent_key}")
    config = CONFIGS[agent_key]

    world = generate_world(seed=seed, n_items=items, horizon_days=horizon)
    at_risk = world.total_at_risk_paise
    runtime = Runtime(
        world,
        config.factory(),
        PolicyEngine(list(DEFAULT_RULES), enabled=config.policy_enabled),
        RazorpayClient(mode="mock", seed=world.seed),
        LLMGateway(backend=None, enabled=config.llm_enabled),
    )
    result = runtime.run()
    result.agent_name = agent_key
    data = result.to_dict()

    # The ledger is the source of truth for the timeline. Everything the page
    # shows is derived from entries that were hash chained during the run, so
    # the dashboard cannot show a number the audit trail does not support.
    events: list[dict[str, Any]] = []
    running_recovered = 0
    for entry in (e.to_dict() for e in runtime.ledger):
        payload = entry["payload"]
        kind = entry["kind"]
        event: dict[str, Any] = {
            "t": entry["at"],
            "kind": kind,
            "item": entry["item_id"],
        }
        if kind == "action":
            action = payload.get("action", {})
            outcome = (payload.get("result") or {}).get("outcome") or {}
            recovered = outcome.get("recovered_paise", 0) or 0
            running_recovered += recovered
            event.update({
                "action": action.get("kind"),
                "channel": action.get("channel"),
                "amount": action.get("amount_paise", 0),
                "p": action.get("p_recover", 0.0),
                "ev": action.get("expected_value_paise", 0),
                "why": action.get("rationale", ""),
                "success": bool(outcome.get("success")),
                "recovered": recovered,
                "cumulative": running_recovered,
                "message": (payload.get("result") or {}).get("message"),
                "reply": payload.get("reply_text"),
                "reply_parsed": (payload.get("reply_parsed") or {}).get("intent"),
            })
        elif kind == "policy_denied":
            decision = payload.get("decision", {})
            event.update({
                "action": (payload.get("action") or {}).get("kind"),
                "rules": [d["rule_id"] for d in decision.get("denials", [])],
                "why": "; ".join(d["reason"] for d in decision.get("denials", []))[:240],
                "cumulative": running_recovered,
            })
        elif kind == "closed":
            event.update({"state": payload.get("state"), "why": payload.get("reason", ""),
                          "cumulative": running_recovered})
        elif kind == "wait":
            event.update({"until": payload.get("until"), "why": payload.get("why", ""),
                          "cumulative": running_recovered})
        events.append(event)

    truncated = len(events) > MAX_EVENTS
    if truncated:
        events = events[:MAX_EVENTS]

    by_method: dict[str, dict[str, int]] = {}
    for item in world.items:
        bucket = by_method.setdefault(item.method.value, {"at_risk": 0, "recovered": 0, "n": 0, "n_rec": 0})
        bucket["at_risk"] += item.amount_paise
        bucket["recovered"] += item.recovered_paise
        bucket["n"] += 1
        if item.recovered_paise > 0:
            bucket["n_rec"] += 1

    return {
        "agent": agent_key,
        "label": CONFIGS[agent_key].label,
        "note": CONFIGS[agent_key].note,
        "seed": seed,
        "items": items,
        "horizon": horizon,
        "start": iso_or_none(world.start),
        "end": iso_or_none(world.end),
        "at_risk_paise": at_risk,
        "summary": data,
        "ledger_head": result.ledger_head,
        "by_method": by_method,
        "events": events,
        "events_truncated": truncated,
    }


def iso_or_none(value):
    from ..clock import iso

    return iso(value) if value else None


class Handler(BaseHTTPRequestHandler):
    server_version = "secondask"

    def log_message(self, fmt_str, *args):  # noqa: A002
        pass  # the default logger writes a line per asset request

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            # The browser navigated away mid-response. Not an error.
            pass

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if route == "/api/agents":
            return self._json(200, {
                "agents": [
                    {"key": k, "label": c.label, "note": c.note, "gated": c.policy_enabled}
                    for k, c in CONFIGS.items()
                ]
            })
        if route == "/api/run":
            query = parse_qs(parsed.query)
            agent = (query.get("agent") or ["secondask"])[0]
            try:
                seed = int((query.get("seed") or ["7"])[0])
                items = max(10, min(3000, int((query.get("items") or ["400"])[0])))
                horizon = max(3, min(60, int((query.get("horizon") or ["21"])[0])))
            except ValueError:
                return self._json(400, {"error": "seed, items and horizon must be integers"})
            if not _RUN_LOCK.acquire(blocking=False):
                return self._json(429, {"error": "a run is already in progress"})
            try:
                return self._json(200, build_run(agent, seed, items, horizon))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            finally:
                _RUN_LOCK.release()

        return self._json(404, {"error": "not found"})

    def _file(self, name: str, content_type: str) -> None:
        path = os.path.join(STATIC, name)
        if not os.path.isfile(path):
            return self._json(404, {"error": f"missing {name}"})
        with open(path, "rb") as handle:
            self._send(200, handle.read(), content_type)


def serve(host: str = "127.0.0.1", port: int = 8420, items: int = 400, horizon: int = 21) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"secondask dashboard on http://{host}:{port}")
    print("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
