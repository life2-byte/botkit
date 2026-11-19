"""Web tools (2): fetch_url, web_search. Public internet only (private/internal addresses are blocked)."""
from __future__ import annotations

import html
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..security import Policy
from ..tool import Tool, clip, tool


def _check_public_host(host: str) -> None:
    """Refuse hosts that resolve to private/loopback/link-local addresses (SSRF protection)."""
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise PermissionError(f"blocked: '{host}' is not a public internet address")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # we follow redirects ourselves, re-checking each hop
        return None


def _fetch(url: str, max_bytes: int = 1_000_000) -> tuple[str, str]:
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(5):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise PermissionError("only http(s) URLs are allowed")
        _check_public_host(parsed.hostname)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; botkit/0.3)"})
        try:
            resp = opener.open(req, timeout=15)
        except urllib.error.HTTPError as e:
            location = e.headers.get("Location") if e.headers else None
            if e.code in (301, 302, 303, 307, 308) and location:
                url = urllib.parse.urljoin(url, location)
                continue
            raise
        with resp:
            kind = resp.headers.get_content_type()
            if not (kind.startswith("text/") or kind.endswith(("json", "xml"))):
                return kind, f"[{kind} content, not shown]"
            charset = resp.headers.get_content_charset() or "utf-8"
            return kind, resp.read(max_bytes).decode(charset, errors="replace")
    raise ValueError("too many redirects")


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?i)<(br|/p|/div|/li|/h\d|/tr)\b[^>]*>", "\n", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


_RESULT_A = re.compile(r'<a\s[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>(.*?)</a>', re.S)


def parse_search_results(page: str, limit: int = 8) -> list[dict[str, str]]:
    """Parse DuckDuckGo's HTML results page into [{title, url, snippet}]."""
    matches = list(_RESULT_A.finditer(page))
    results = []
    for i, m in enumerate(matches[:limit]):
        href_m = re.search(r'href="([^"]*)"', m.group(0))
        if not href_m:
            continue
        href = html.unescape(href_m.group(1))
        if "uddg=" in href:  # DuckDuckGo wraps the real link
            query = urllib.parse.urlparse(href if href.startswith("http") else "https:" + href).query
            href = urllib.parse.parse_qs(query).get("uddg", [href])[0]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page)
        sn = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div|span)>', page[m.end():end], re.S)
        results.append(
            {"title": _html_to_text(m.group(1)), "url": href, "snippet": _html_to_text(sn.group(1)) if sn else ""}
        )
    return results


def make(policy: Policy, sandbox=None) -> list[Tool]:
    if not policy.allow_network:
        return []

    @tool
    def fetch_url(url: str, max_chars: int = 15000) -> str:
        """Download a public web page or API response and return its text. Private/internal addresses are blocked.

        Args:
            url: A public http(s) URL.
            max_chars: Maximum characters to return.
        """
        kind, body = _fetch(url)
        if kind == "text/html":
            body = _html_to_text(body)
        return clip(body, max_chars)

    @tool
    def web_search(query: str, max_results: int = 6) -> str:
        """Search the web (DuckDuckGo) and return titles, URLs and snippets. Use fetch_url to read a result.

        Args:
            query: What to search for.
            max_results: How many results to return (1-10).
        """
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        _, page = _fetch(url)
        results = parse_search_results(page, max(1, min(int(max_results), 10)))
        if not results:
            return "No results (the search provider may be blocking automated requests). Try fetch_url on a known site."
        return "\n\n".join(
            f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}".rstrip() for i, r in enumerate(results, 1)
        )

    return [fetch_url, web_search]
