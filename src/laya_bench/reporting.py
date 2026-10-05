"""Render arbitrary benchmark result objects without inventing absent measurements."""
from __future__ import annotations

import html
import json
from typing import Any, Mapping


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return str(value)


def _markdown_value(value: Any) -> str:
    text = _display(value)
    if isinstance(value, (dict, list, tuple)):
        return "```json\n" + text + "\n```"
    return text.replace("|", "\\|").replace("\n", "<br>")


def render_markdown(result: Any, title: str = "Laya benchmark report") -> str:
    """Render nested JSON-compatible results as a compact Markdown report."""
    lines = [f"# {title}", ""]

    def visit(value: Any, heading_level: int, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            if key is not None:
                lines.extend(["#" * min(heading_level, 6) + " " + str(key), ""])
            scalars = [(k, v) for k, v in value.items() if not isinstance(v, (Mapping, list, tuple))]
            nested = [(k, v) for k, v in value.items() if isinstance(v, (Mapping, list, tuple))]
            if scalars:
                lines.extend(["| Metric | Value |", "|---|---|"])
                lines.extend(f"| {str(k).replace('|', '\\|')} | {_markdown_value(v)} |" for k, v in scalars)
                lines.append("")
            for k, v in nested:
                visit(v, heading_level + 1, str(k))
        elif isinstance(value, (list, tuple)):
            if key is not None:
                lines.extend(["#" * min(heading_level, 6) + " " + str(key), ""])
            if not value:
                lines.extend(["_(no records)_", ""])
            for i, item in enumerate(value, 1):
                if isinstance(item, (Mapping, list, tuple)):
                    visit(item, heading_level + 1, f"Item {i}")
                else:
                    lines.append(f"- {_markdown_value(item)}")
            lines.append("")
        else:
            if key is not None:
                lines.extend(["#" * min(heading_level, 6) + " " + str(key), ""])
            lines.extend([_markdown_value(value), ""])

    visit(result, 2)
    return "\n".join(lines).rstrip() + "\n"


def render_html(result: Any, title: str = "Laya benchmark report") -> str:
    """Render arbitrary JSON-compatible results as escaped, readable HTML."""
    def node(value: Any) -> str:
        if isinstance(value, Mapping):
            if not value:
                return '<p class="empty">No fields</p>'
            rows = []
            for key, item in value.items():
                rendered = node(item)
                rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{rendered}</td></tr>")
            return "<table><tbody>" + "".join(rows) + "</tbody></table>"
        if isinstance(value, (list, tuple)):
            if not value:
                return '<p class="empty">No records</p>'
            return "<ol>" + "".join(f"<li>{node(item)}</li>" for item in value) + "</ol>"
        if value is None:
            return '<span class="missing">—</span>'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                return '<span class="missing">—</span>'
        return html.escape(_display(value))

    return ("<!doctype html><html><head><meta charset=\"utf-8\"><title>" + html.escape(title) +
            "</title><style>body{font:15px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}"
            "table{border-collapse:collapse;margin:.4rem 0 1.2rem;width:100%}th,td{text-align:left;border:1px solid #ddd;padding:.45rem;vertical-align:top}"
            "th{background:#f4f5f7}ol{padding-left:1.5rem}.missing,.empty{color:#777}</style></head><body><h1>" +
            html.escape(title) + "</h1>" + node(result) + "</body></html>")


def render_report(result: Any, format: str = "markdown", title: str = "Laya benchmark report") -> str:
    """Convenience dispatcher for Markdown or HTML output."""
    normalized = format.lower()
    if normalized in ("markdown", "md"):
        return render_markdown(result, title)
    if normalized == "html":
        return render_html(result, title)
    raise ValueError("format must be 'markdown' or 'html'")
