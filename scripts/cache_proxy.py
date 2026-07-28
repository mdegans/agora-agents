#!/usr/bin/env python3
"""Transparent blallama proxy + `/probe` recorder for cache-hit analysis.

Sits between the seed runner and blallama, forwarding every path
verbatim (`/api/tags` for model discovery, `/v1/messages` for
inference) while recording, for each inference request:

  * the request body (messages, system, tools) so a cache miss can be
    diffed against the request that should have primed it,
  * the response `usage` — `input_tokens`, `cache_read_input_tokens`,
    `cache_creation_input_tokens`,
  * the **deficit** metric, and
  * the joined `/probe` SSE session for that response id.

Why the deficit rather than watching `cache_read` climb: a legitimate
new-session start and a prefix-cache collapse produce an *identical*
stats line (both read the `system + tools` floor). They are
indistinguishable from one request alone. What separates them is
comparing against the **previous request's `input_tokens`**:

    deficit = prev.input_tokens - this.cache_read

Negative is healthy — the cache also holds the prior turn's output.
Positive means the server failed to reuse something it already had.
`cache_read` can *increase* while the deficit is large and positive,
which is why "cache reads should always increase" is necessary but not
sufficient as a health check. See
`memory/project_blallama_cache_miss_analysis_2026_05_12.md`.

The `/probe` join is the complementary instrument: `session_start.id`
matches the `Message.id` returned by `/v1/messages`, and `ctx.n_cur` at
the first generated token equals that request's `input_tokens` exactly.
When they disagree, the server and the sampler disagree about how much
prefix was actually resident.

By default only a *summary* of each probe session is kept (first-token
`n_cur`, token count, entropy stats). `--full-probe` retains every
token's top-K, which is order-100 floats per token — useful for a
single-request post-mortem, far too heavy for a seed run.

Stdlib only, by design: this runs as an operational sidecar and must not
depend on the environment it is instrumenting.

Usage:

    ./cache_proxy.py --upstream http://192.168.0.123:11435 --port 11500
    # then point the runner at  blallama://localhost:11500

Output (in --outdir, default ./cache_telemetry):

    requests.jsonl   one record per /v1/messages round-trip
    COLLAPSE_<n>_<id>.json   full request+response dump whenever the
                             deficit exceeds --collapse-threshold
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers that must not be forwarded verbatim (RFC 9110 hop-by-hop).
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Recorder:
    """Accumulates request records and joins them against probe sessions.

    Thread-safe: the proxy handler threads and the probe reader thread
    both touch this.
    """

    def __init__(self, outdir: str, collapse_threshold: int, full_probe: bool):
        self.outdir = outdir
        self.collapse_threshold = collapse_threshold
        self.full_probe = full_probe
        self.lock = threading.Lock()
        self.seq = 0
        # Previous request's input_tokens, globally and per model. The
        # runner keeps concurrency=1 on blallama, so the global sequence
        # is a true chain; per-model is kept for interleaved cohorts.
        self.prev_input_tokens: int | None = None
        self.prev_input_tokens_by_model: dict[str, int] = {}
        # The previous request *body* and its record, kept so a COLLAPSE
        # dump is a self-contained reproducer. A stall is a property of a
        # *pair* of requests — the one that primed the cache and the one
        # that failed to reuse it — so the collapsing request alone
        # replays as an ordinary cold start and reproduces nothing.
        # One deep: concurrency is 1 on blallama, so the global sequence
        # is a true chain.
        self.prev_request: dict | None = None
        self.prev_record: dict | None = None
        # Probe sessions keyed by message id, awaiting their request
        # record (the SSE session_start usually lands *before* the
        # synchronous /v1/messages response returns).
        self.probe_sessions: dict[str, dict] = {}
        # Request records awaiting their probe session, keyed by id.
        self.pending: dict[str, dict] = {}
        os.makedirs(outdir, exist_ok=True)
        self.requests_path = os.path.join(outdir, "requests.jsonl")

    # -- probe side ---------------------------------------------------

    def probe_session_start(self, sid: str, model: str) -> None:
        with self.lock:
            self.probe_sessions[sid] = {
                "model": model,
                "started_at": now(),
                "tokens": [],
                "first_n_cur": None,
                "entropies": [],
            }

    def probe_token(self, sid: str, ctx: dict) -> None:
        with self.lock:
            sess = self.probe_sessions.get(sid)
            if sess is None:
                # session_start missed (we connected mid-generation).
                sess = self.probe_sessions.setdefault(
                    sid,
                    {
                        "model": None,
                        "started_at": now(),
                        "tokens": [],
                        "first_n_cur": None,
                        "entropies": [],
                    },
                )
            if sess["first_n_cur"] is None:
                sess["first_n_cur"] = ctx.get("n_cur")
            snap = ctx.get("snapshot")
            if isinstance(snap, dict) and "entropy" in snap:
                sess["entropies"].append(snap["entropy"])
            if self.full_probe:
                sess["tokens"].append(ctx)
            else:
                # Keep only the count; the top-K is the heavy part.
                sess["tokens"].append(None)

    def probe_session_end(self, sid: str) -> None:
        with self.lock:
            sess = self.probe_sessions.get(sid)
            if sess is not None:
                sess["ended_at"] = now()
            record = self.pending.pop(sid, None)
        if record is not None:
            # The request record arrived first and was waiting on us.
            self._finalize(record, sid)

    def _summarize_probe(self, sid: str) -> dict | None:
        """Caller must hold the lock."""
        sess = self.probe_sessions.pop(sid, None)
        if sess is None:
            return None
        entropies = sess.get("entropies") or []
        summary = {
            "model": sess.get("model"),
            "started_at": sess.get("started_at"),
            "ended_at": sess.get("ended_at"),
            "first_n_cur": sess.get("first_n_cur"),
            "n_tokens": len(sess.get("tokens") or []),
        }
        if entropies:
            summary["entropy_mean"] = round(statistics.fmean(entropies), 4)
            summary["entropy_max"] = round(max(entropies), 4)
            summary["entropy_min"] = round(min(entropies), 4)
        if self.full_probe:
            summary["tokens"] = sess.get("tokens")
        return summary

    # -- request side -------------------------------------------------

    def record(self, req_body: bytes, resp_body: bytes, elapsed_ms: float) -> None:
        """Record one /v1/messages round-trip.

        Waits briefly for the probe session to close so the join lands in
        the same record; if it doesn't, the record is written without it
        rather than blocking the run.
        """
        try:
            req = json.loads(req_body) if req_body else {}
        except (ValueError, UnicodeDecodeError):
            req = {"_unparseable": True}
        try:
            resp = json.loads(resp_body) if resp_body else {}
        except (ValueError, UnicodeDecodeError):
            resp = {"_unparseable": True}

        # An error response carries no usage and no message id. Its
        # "deficit" would be computed against a cache_read of 0, which is
        # meaningless — record it as an error and leave the deficit chain
        # untouched so the next real request still compares against the
        # last real one.
        if isinstance(resp, dict) and resp.get("type") == "error":
            self._record_error(req, resp, elapsed_ms)
            return

        usage = resp.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_creation = usage.get("cache_creation_input_tokens") or 0
        model = resp.get("model") or req.get("model")
        msg_id = resp.get("id")

        with self.lock:
            self.seq += 1
            seq = self.seq
            prev_global = self.prev_input_tokens
            prev_model = self.prev_input_tokens_by_model.get(model) if model else None
            if isinstance(input_tokens, int):
                self.prev_input_tokens = input_tokens
                if model:
                    self.prev_input_tokens_by_model[model] = input_tokens

        # deficit = prev.input_tokens - this.cache_read. Positive means
        # the server failed to reuse a prefix it had already built.
        deficit = None if prev_global is None else prev_global - cache_read
        deficit_model = None if prev_model is None else prev_model - cache_read

        record = {
            "seq": seq,
            "at": now(),
            "id": msg_id,
            "model": model,
            "elapsed_ms": round(elapsed_ms, 1),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": usage.get("output_tokens"),
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
            "prev_input_tokens": prev_global,
            "deficit": deficit,
            "prev_input_tokens_same_model": prev_model,
            "deficit_same_model": deficit_model,
            "n_messages": len(req.get("messages") or []),
            "has_tools": bool(req.get("tools")),
            "n_tools": len(req.get("tools") or []),
            "stop_reason": resp.get("stop_reason"),
        }

        if msg_id:
            # Give the probe stream a moment to emit session_end. The
            # SSE side usually finishes first, but the ordering is not
            # guaranteed and a synchronous response can beat it.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with self.lock:
                    sess = self.probe_sessions.get(msg_id)
                    if sess is not None and "ended_at" in sess:
                        break
                time.sleep(0.02)

        self._finalize(record, msg_id, req, resp)

    def _record_error(self, req: dict, resp: dict, elapsed_ms: float) -> None:
        """Log an upstream error without disturbing the deficit chain."""
        err = resp.get("error") or {}
        with self.lock:
            self.seq += 1
            seq = self.seq
        record = {
            "seq": seq,
            "at": now(),
            "id": None,
            "model": req.get("model"),
            "elapsed_ms": round(elapsed_ms, 1),
            "error": {
                "type": err.get("type"),
                "code": err.get("code"),
                "message": err.get("message"),
            },
            "n_messages": len(req.get("messages") or []),
            "has_tools": bool(req.get("tools")),
        }
        with self.lock:
            with open(self.requests_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        path = os.path.join(self.outdir, f"ERROR_{seq:04d}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"record": record, "request": req, "response": resp}, fh, indent=2)
        msg = (err.get("message") or "")[:120]
        print(
            f"[{seq:04d}] {record['model']} ERROR {err.get('code')} {msg}\n"
            f"        wrote {path}",
            flush=True,
        )

    def _finalize(
        self,
        record: dict,
        msg_id: str | None,
        req: dict | None = None,
        resp: dict | None = None,
    ) -> None:
        with self.lock:
            probe = self._summarize_probe(msg_id) if msg_id else None
            record["probe"] = probe
            # ctx.n_cur at the first generated token should equal this
            # request's input_tokens. Disagreement means the server and
            # the sampler disagree about resident prefix length.
            if probe and isinstance(probe.get("first_n_cur"), int):
                it = record["usage"].get("input_tokens")
                if isinstance(it, int):
                    record["n_cur_vs_input_tokens"] = probe["first_n_cur"] - it
            line = json.dumps(record, separators=(",", ":"))
            with open(self.requests_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        deficit = record.get("deficit")
        # A request with a single message is a *fresh conversation* — a new
        # agent starting its cycle. Its cache_read legitimately falls back
        # to the system+tools floor, which produces a large positive
        # deficit against whatever the previous agent was doing. That is
        # indistinguishable from a real prefix collapse by the stats line
        # alone, and flagging it cries wolf at every agent boundary.
        # Only a deficit *within* an ongoing conversation is a collapse.
        agent_boundary = record.get("n_messages") == 1
        flagged = (
            isinstance(deficit, int)
            and deficit > self.collapse_threshold
            and not agent_boundary
        )
        record["agent_boundary"] = agent_boundary
        if flagged and req is not None:
            self._dump_collapse(record, req, resp)
        # Rotate *after* the dump so the dump sees the true predecessor.
        with self.lock:
            self.prev_request = req
            self.prev_record = record

        d = "n/a" if deficit is None else f"{deficit:+d}"
        marker = "  <-- COLLAPSE" if flagged else ""
        print(
            f"[{record['seq']:04d}] {record.get('model')} "
            f"in={record['usage'].get('input_tokens')} "
            f"read={record['usage'].get('cache_read_input_tokens')} "
            f"create={record['usage'].get('cache_creation_input_tokens')} "
            f"deficit={d}{marker}",
            flush=True,
        )

    def _dump_collapse(self, record: dict, req: dict, resp: dict | None) -> None:
        name = f"COLLAPSE_{record['seq']:04d}_{record.get('id') or 'unknown'}.json"
        path = os.path.join(self.outdir, name)
        with self.lock:
            prev_request = self.prev_request
            prev_record = self.prev_record
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "record": record,
                    "request": req,
                    "response": resp,
                    # The primer. Replay `prev_request` then `request`
                    # against a fresh server to reproduce the stall;
                    # `request` alone is just a cold start.
                    "prev_record": prev_record,
                    "prev_request": prev_request,
                },
                fh,
                indent=2,
            )
        print(f"        wrote {path}", flush=True)


def make_handler(upstream: str, recorder: Recorder):
    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A003 - silence access log
            pass

        def _proxy(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in HOP_BY_HOP and k.lower() != "host"
            }
            # Ask upstream for identity encoding so we can parse usage
            # without pulling in a decompressor.
            headers["Accept-Encoding"] = "identity"

            url = upstream.rstrip("/") + self.path
            req = urllib.request.Request(
                url, data=body or None, headers=headers, method=method
            )

            started = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=1800) as up:
                    status = up.status
                    resp_headers = up.headers
                    resp_body = up.read()
            except urllib.error.HTTPError as e:
                status = e.code
                resp_headers = e.headers
                resp_body = e.read()
            except Exception as e:  # upstream unreachable
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                payload = json.dumps({"error": f"proxy upstream failure: {e}"}).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            elapsed_ms = (time.monotonic() - started) * 1000

            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() in HOP_BY_HOP or k.lower() == "content-length":
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

            if method == "POST" and self.path.rstrip("/").endswith("/v1/messages"):
                try:
                    recorder.record(body, resp_body, elapsed_ms)
                except Exception as e:  # never let telemetry break the run
                    print(f"!! recorder error: {e}", file=sys.stderr, flush=True)

        def do_POST(self):  # noqa: N802
            self._proxy("POST")

        def do_GET(self):  # noqa: N802
            self._proxy("GET")

        def do_DELETE(self):  # noqa: N802
            self._proxy("DELETE")

    return ProxyHandler


def probe_reader(probe_url: str, recorder: Recorder, stop: threading.Event) -> None:
    """Consume the `/probe` SSE stream, reconnecting until told to stop."""
    while not stop.is_set():
        try:
            req = urllib.request.Request(
                probe_url, headers={"Accept": "text/event-stream"}
            )
            # No read timeout: SSE connections are long-lived and idle
            # between generations. A timeout here is the classic bug.
            with urllib.request.urlopen(req) as stream:
                print(f"probe: connected to {probe_url}", flush=True)
                for raw in stream:
                    if stop.is_set():
                        return
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        evt = json.loads(payload)
                    except ValueError:
                        continue
                    kind = evt.get("event")
                    sid = evt.get("id")
                    if not sid:
                        continue
                    if kind == "session_start":
                        recorder.probe_session_start(sid, evt.get("model"))
                    elif kind == "token":
                        recorder.probe_token(sid, evt.get("ctx") or {})
                    elif kind == "session_end":
                        recorder.probe_session_end(sid)
        except Exception as e:
            if stop.is_set():
                return
            print(f"probe: disconnected ({e}); retrying in 3s", flush=True)
            stop.wait(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--upstream",
        default="http://192.168.0.123:11435",
        help="blallama base URL (default: %(default)s)",
    )
    ap.add_argument("--port", type=int, default=11500, help="listen port")
    ap.add_argument("--host", default="127.0.0.1", help="listen address")
    ap.add_argument("--outdir", default="cache_telemetry", help="output directory")
    ap.add_argument(
        "--collapse-threshold",
        type=int,
        default=64,
        help="dump a COLLAPSE_*.json when deficit exceeds this (default: %(default)s)",
    )
    ap.add_argument(
        "--full-probe",
        action="store_true",
        help="retain every token's top-K (very large; single-request post-mortems only)",
    )
    ap.add_argument(
        "--no-probe", action="store_true", help="skip the /probe SSE subscription"
    )
    args = ap.parse_args()

    recorder = Recorder(args.outdir, args.collapse_threshold, args.full_probe)
    stop = threading.Event()

    if not args.no_probe:
        probe_url = args.upstream.rstrip("/") + "/probe"
        threading.Thread(
            target=probe_reader, args=(probe_url, recorder, stop), daemon=True
        ).start()

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.upstream, recorder)
    )
    print(
        f"cache_proxy: {args.host}:{args.port} -> {args.upstream}\n"
        f"             recording to {recorder.requests_path}\n"
        f"             point the runner at blallama://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
