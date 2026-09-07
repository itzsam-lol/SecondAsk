"""FastAPI transport.

Deliberately a separate module, and deliberately **without**
``from __future__ import annotations``.

That import turns every annotation into a string. FastAPI hands its annotations
to pydantic, which resolves those strings against the *module* globals. With the
FastAPI names imported inside a function, as they were when this lived in
``api.py``, ``Request`` is not a module global and pydantic raises::

    PydanticUndefinedAnnotation: name 'Request' is not defined

The failure only appears when FastAPI is actually installed, which is exactly
when it matters and exactly when a zero-dependency test run will not catch it.
Isolating the framework here keeps the postponed-annotations style everywhere
else and lets ``api.py`` import this behind a plain try/except.

The handlers are thin on purpose. Every security decision lives in
``IngestionService``, so the FastAPI path and the stdlib path cannot drift.
"""

import json
from typing import Optional

from fastapi import FastAPI, Header, Request, Response

from .api import IngestionService


def create_app(service: Optional[IngestionService] = None) -> FastAPI:
    svc = service or IngestionService()
    app = FastAPI(
        title="SecondAsk ingestion",
        version="1.0",
        description=(
            "Webhook and inbound message ingestion for the SecondAsk recovery agent. "
            "Webhooks are HMAC verified before parsing; inbound messages can never "
            "settle an item."
        ),
    )

    @app.post("/webhooks/razorpay")
    async def razorpay_webhook(request: Request, x_razorpay_signature: str = Header(default="")):
        # The raw bytes, not a parsed model.
        #
        # Declaring a pydantic body model here would be the idiomatic FastAPI
        # thing and would break signature verification: the framework would parse
        # and re-serialise the payload, and the bytes that reach the verifier
        # would no longer be the bytes Razorpay signed.
        body = await request.body()
        status, payload = svc.handle_webhook(body, x_razorpay_signature or None)
        return Response(json.dumps(payload), status_code=status, media_type="application/json")

    @app.post("/inbound/message")
    async def inbound_message(request: Request):
        status, payload = svc.handle_inbound(await request.body())
        return Response(json.dumps(payload), status_code=status, media_type="application/json")

    @app.get("/health")
    async def health():
        status, payload = svc.health()
        return Response(json.dumps(payload), status_code=status, media_type="application/json")

    @app.get("/metrics")
    async def metrics():
        return Response(svc.metrics.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/metrics.json")
    async def metrics_json():
        return Response(json.dumps(svc.metrics_json()), media_type="application/json")

    app.state.service = svc
    return app
