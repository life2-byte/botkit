"""Document tools (3): create_docx, create_xlsx, create_pptx. Needs: pip install 'botkit[docs]'."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..security import Policy
from ..tool import Tool, tool

OFFICE_EXT = {".docx", ".xlsx", ".pptx"}


def _need(pip_name: str) -> ImportError:
    return ImportError(f"This tool needs '{pip_name}'. Install it with: pip install 'botkit[docs]'")


# ---------------------------------------------------------------- reading
def read_office(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".docx":
        try:
            import docx
        except ImportError:
            raise _need("python-docx") from None
        d = docx.Document(str(p))
        parts = [para.text for para in d.paragraphs]
        for table in d.tables:
            parts += [" | ".join(c.text for c in row.cells) for row in table.rows]
        return "\n".join(parts)
    if ext == ".xlsx":
        try:
            import openpyxl
        except ImportError:
            raise _need("openpyxl") from None
        wb = openpyxl.load_workbook(str(p))
        out = []
        for ws in wb.worksheets:
            out.append(f"## {ws.title}")
            for row in ws.iter_rows(max_row=500, values_only=True):
                out.append(" | ".join("" if c is None else str(c) for c in row))
        return "\n".join(out)
    try:
        from pptx import Presentation
    except ImportError:
        raise _need("python-pptx") from None
    out = []
    for i, slide in enumerate(Presentation(str(p)).slides, 1):
        out.append(f"## Slide {i}")
        for shape in slide.shapes:
            if shape.has_text_frame:
                out.append(shape.text_frame.text)
    return "\n".join(out)


# ---------------------------------------------------------------- docx content
def _runs(paragraph: Any, text: str) -> None:
    for i, chunk in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if chunk:
            paragraph.add_run(chunk).bold = i % 2 == 1


def _fill_docx(doc: Any, content: str) -> None:
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        if table_rows:
            cols = max(len(r) for r in table_rows)
            table = doc.add_table(rows=len(table_rows), cols=cols)
            table.style = "Table Grid"
            for r, row in enumerate(table_rows):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
            table_rows.clear()

    for line in content.splitlines():
        s = line.rstrip()
        if s.strip().startswith("|"):
            cells = [c.strip() for c in s.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                table_rows.append(cells)
            continue
        flush_table()
        if not s.strip():
            continue
        if m := re.match(r"^(#{1,4})\s+(.*)", s):
            doc.add_heading(m.group(2), level=len(m.group(1)))
        elif m := re.match(r"^\s*[-*]\s+(.*)", s):
            _runs(doc.add_paragraph(style="List Bullet"), m.group(1))
        elif m := re.match(r"^\s*\d+[.)]\s+(.*)", s):
            _runs(doc.add_paragraph(style="List Number"), m.group(1))
        else:
            _runs(doc.add_paragraph(), s)
    flush_table()


def make(policy: Policy, sandbox=None) -> list[Tool]:
    def target(path: str, ext: str) -> Path:
        return policy.resolve(path if path.lower().endswith(ext) else path + ext)

    def rel(p: Path) -> Path:
        return p.relative_to(policy.workdir.resolve())

    @tool
    def create_docx(path: str, title: str, content: str) -> str:
        """Create a Word document. Content uses simple markdown: # headings, - bullets, 1. numbered, **bold**, | tables |.

        Args:
            path: Output path relative to the workspace, e.g. "report.docx".
            title: Document title.
            content: The body in simple markdown.
        """
        try:
            import docx
        except ImportError:
            raise _need("python-docx") from None
        p = target(path, ".docx")
        doc = docx.Document()
        doc.add_heading(title, level=0)
        _fill_docx(doc, content)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(p))
        return f"Created {rel(p)}"

    @tool
    def create_xlsx(path: str, sheets: dict) -> str:
        """Create an Excel workbook. Strings starting with "=" become formulas; the first row of each sheet is bold.

        Args:
            path: Output path relative to the workspace, e.g. "budget.xlsx".
            sheets: Mapping of sheet name to a list of rows, each row a list of cell values.
        """
        try:
            import openpyxl
            from openpyxl.styles import Font
        except ImportError:
            raise _need("openpyxl") from None
        if not isinstance(sheets, dict) or not sheets:
            raise ValueError("sheets must be a non-empty object: {sheet name: [[row], [row]]}")
        p = target(path, ".xlsx")
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        for name, rows in sheets.items():
            if not isinstance(rows, list) or not all(isinstance(r, (list, tuple)) for r in rows):
                raise ValueError(f"sheet '{name}' must be a list of rows (lists)")
            ws = wb.create_sheet(title=(str(name) or "Sheet")[:31])
            for row in rows:
                ws.append(list(row))
            if rows:
                for cell in ws[1]:
                    cell.font = Font(bold=True)
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 50)
        p.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(p))
        return f"Created {rel(p)}"

    @tool
    def create_pptx(path: str, slides: list[dict]) -> str:
        """Create a PowerPoint deck. Each slide is an object with "title", optional "bullets" (list of strings),
        optional "subtitle" (for a title slide) and optional "notes" (speaker notes).

        Args:
            path: Output path relative to the workspace, e.g. "deck.pptx".
            slides: The slides, in order.
        """
        try:
            from pptx import Presentation
        except ImportError:
            raise _need("python-pptx") from None
        if not slides:
            raise ValueError("slides must not be empty")
        p = target(path, ".pptx")
        prs = Presentation()
        for s in slides:
            bullets = [str(b) for b in s.get("bullets") or []]
            if s.get("subtitle") and not bullets:
                slide = prs.slides.add_slide(prs.slide_layouts[0])
                slide.shapes.title.text = str(s.get("title", ""))
                slide.placeholders[1].text = str(s["subtitle"])
            else:
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = str(s.get("title", ""))
                frame = slide.placeholders[1].text_frame
                frame.text = bullets[0] if bullets else ""
                for b in bullets[1:]:
                    frame.add_paragraph().text = b
            if s.get("notes"):
                slide.notes_slide.notes_text_frame.text = str(s["notes"])
        p.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(p))
        return f"Created {rel(p)} ({len(slides)} slides)"

    return [create_docx, create_xlsx, create_pptx]
