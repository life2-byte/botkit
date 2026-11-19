"""A minimal PostgREST look-alike (enough for SupabaseStorage tests): filters, order, limit, returning."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse


class FakePostgrest:
    def __init__(self, tables=("botkit_messages", "botkit_facts", "orders"), key="service-key"):
        self.tables = {t: [] for t in tables}
        self.key, self.next_id, self.requests = key, 1, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None):
                body = b"" if obj is None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _match(self, rows, query):
                out = rows
                for col, vals in query.items():
                    if col in ("select", "order", "limit"):
                        continue
                    op, _, val = vals[0].partition(".")
                    cmp = {
                        "eq": lambda a, b: str(a) == b, "gt": lambda a, b: float(a) > float(b),
                        "lt": lambda a, b: float(a) < float(b),
                    }[op]
                    out = [r for r in out if col in r and cmp(r[col], unquote(val))]
                return out

            def _handle(self, method):
                u = urlparse(self.path)
                query = parse_qs(u.query)
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                outer.requests.append((method, u.path, query, {k.lower(): v for k, v in self.headers.items()}))
                if self.headers.get("apikey") != outer.key or self.headers.get("Authorization") != f"Bearer {outer.key}":
                    return self._send(401, {"message": "Invalid API key"})
                m = re.fullmatch(r"/rest/v1/(\w+)", u.path)
                if not m:
                    return self._send(404, {"message": "bad path"})
                name = m.group(1)
                if name not in outer.tables:
                    return self._send(404, {"code": "PGRST205", "message": f"Could not find the table 'public.{name}' in the schema cache"})
                rows, ret = outer.tables[name], "return=representation" in (self.headers.get("Prefer") or "")
                if method == "POST":
                    items = body if isinstance(body, list) else [body]
                    new = []
                    for it in items:
                        row = {"id": outer.next_id, **it}
                        outer.next_id += 1
                        rows.append(row)
                        new.append(row)
                    return self._send(201, new if ret else None)
                matched = self._match(rows, query)
                if method == "GET":
                    if "order" in query:
                        col, _, direction = query["order"][0].partition(".")
                        matched = sorted(matched, key=lambda r: r[col], reverse=direction == "desc")
                    if "limit" in query:
                        matched = matched[: int(query["limit"][0])]
                    cols = query.get("select", ["*"])[0]
                    if cols != "*":
                        matched = [{c: r[c] for c in cols.split(",")} for r in matched]
                    return self._send(200, matched)
                if method == "DELETE":
                    outer.tables[name] = [r for r in rows if r not in matched]
                    return self._send(200 if ret else 204, matched if ret else None)
                if method == "PATCH":
                    for r in matched:
                        r.update(body)
                    return self._send(200, matched if ret else None)
                self._send(405, {"message": "nope"})

            def do_GET(self): self._handle("GET")
            def do_POST(self): self._handle("POST")
            def do_DELETE(self): self._handle("DELETE")
            def do_PATCH(self): self._handle("PATCH")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
