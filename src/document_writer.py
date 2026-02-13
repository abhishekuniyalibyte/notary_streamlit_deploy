"""
Document Writer Module

Handles generation of fixed output documents (PDF, DOCX, RTF) with corrected text.
Moved from chatbot.py to reduce file size and improve modularity.
"""

import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

try:
    import streamlit as st
except ImportError:
    # Fallback for non-Streamlit environments
    class st:
        class session_state:
            @staticmethod
            def get(key, default=None):
                return default

from src.phase4_text_extraction import TextNormalizer


def build_fixed_outputs_for_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate fixed output artifacts from the latest extraction in results.
    Non-destructive: does not modify original uploaded files.
    """
    file_results = results.get("file_results") or []
    extraction_by_path = results.get("_extraction_by_path") or {}
    fixed_dir_errors: List[str] = []
    fixed_dir_base = Path("output").resolve()
    fixed_dir_base.mkdir(parents=True, exist_ok=True)
    case_id = (
        results.get("_case_id")
        or st.session_state.get("chat_case_id")
        or st.session_state.get("case_id")
    )
    case_suffix = f"chat_case_{case_id}" if case_id else datetime.now().strftime("case_%Y%m%d_%H%M%S")
    fixed_dir_root = (fixed_dir_base / "fixed" / case_suffix).resolve()
    fixed_dir_root.mkdir(parents=True, exist_ok=True)

    def write_pdf_from_text(output_path: Path, text: str) -> None:
        # Preferred: use reportlab if installed (better wrapping + encoding).
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.lib.utils import simpleSplit
            from reportlab.pdfgen import canvas

            text = text or ""
            if not text.strip():
                text = "No se pudo extraer texto del documento."

            page_width, page_height = A4
            margin = 18 * mm
            font_name = "Times-Roman"
            font_size = 11
            line_height = 14

            c = canvas.Canvas(str(output_path), pagesize=A4)
            c.setTitle(output_path.stem)
            c.setAuthor("Notaria")
            c.setFont(font_name, font_size)

            y = page_height - margin
            max_width = page_width - (2 * margin)

            for raw_line in text.splitlines():
                line = raw_line.rstrip("\n")
                if not line.strip():
                    y -= line_height
                    if y < margin:
                        c.showPage()
                        c.setFont(font_name, font_size)
                        y = page_height - margin
                    continue

                for wrapped in simpleSplit(line, font_name, font_size, max_width):
                    if y < margin:
                        c.showPage()
                        c.setFont(font_name, font_size)
                        y = page_height - margin
                    c.drawString(margin, y, wrapped)
                    y -= line_height

            c.save()
            return
        except Exception:
            pass

        # Fallback: minimal PDF writer (no external deps). Uses standard Helvetica + WinAnsi.
        import io as _io

        def escape_pdf_literal(value: bytes) -> bytes:
            return (
                value.replace(b"\\", b"\\\\")
                .replace(b"(", b"\\(")
                .replace(b")", b"\\)")
            )

        def wrap_line(value: str, max_chars: int) -> List[str]:
            if not value:
                return [""]
            stripped = value.strip()
            if not stripped:
                return [""]
            words = stripped.split()
            lines: List[str] = []
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if len(candidate) <= max_chars:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                if len(word) > max_chars:
                    for idx in range(0, len(word), max_chars):
                        lines.append(word[idx : idx + max_chars])
                    current = ""
                else:
                    current = word
            if current:
                lines.append(current)
            return lines or [""]

        text = text or ""
        if not text.strip():
            text = "No se pudo extraer texto del documento."

        page_width = 595  # A4 points
        page_height = 842
        margin = 50
        font_size = 11
        leading = 14
        max_lines = max(1, int((page_height - 2 * margin) // leading))
        max_chars = max(20, int((page_width - 2 * margin) / (font_size * 0.6)))

        wrapped_lines: List[str] = []
        for raw in text.splitlines():
            if not raw.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(wrap_line(raw, max_chars))

        pages: List[List[str]] = [
            wrapped_lines[i : i + max_lines]
            for i in range(0, len(wrapped_lines), max_lines)
        ]
        if not pages:
            pages = [["No se pudo extraer texto del documento."]]

        font_obj = 3 + len(pages)
        first_content_obj = font_obj + 1
        total_objs = 2 + len(pages) + 1 + len(pages)

        def content_stream_for_page(lines: List[str]) -> bytes:
            start_y = page_height - margin - font_size
            out = _io.BytesIO()
            out.write(b"BT\n")
            out.write(f"/F1 {font_size} Tf\n".encode("ascii"))
            out.write(f"1 0 0 1 {margin} {start_y} Tm\n".encode("ascii"))
            for line in lines:
                if not line:
                    out.write(f"0 -{leading} Td\n".encode("ascii"))
                    continue
                encoded = line.encode("cp1252", errors="replace")
                out.write(b"(" + escape_pdf_literal(encoded) + b") Tj\n")
                out.write(f"0 -{leading} Td\n".encode("ascii"))
            out.write(b"ET\n")
            return out.getvalue()

        objects: List[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

        kids_refs = " ".join(f"{3 + i} 0 R" for i in range(len(pages)))
        objects.append(f"<< /Type /Pages /Kids [{kids_refs}] /Count {len(pages)} >>".encode("ascii"))

        for i in range(len(pages)):
            page_obj = 3 + i
            content_obj = first_content_obj + i
            objects.append(
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
                    f"/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>"
                ).encode("ascii")
            )

        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        for page_lines in pages:
            stream = content_stream_for_page(page_lines)
            objects.append(
                b"<< /Length "
                + str(len(stream)).encode("ascii")
                + b" >>\nstream\n"
                + stream
                + b"endstream"
            )

        pdf = _io.BytesIO()
        pdf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0] * (total_objs + 1)

        for idx, obj in enumerate(objects, start=1):
            offsets[idx] = pdf.tell()
            pdf.write(f"{idx} 0 obj\n".encode("ascii"))
            pdf.write(obj)
            pdf.write(b"\nendobj\n")

        xref_offset = pdf.tell()
        pdf.write(f"xref\n0 {total_objs + 1}\n".encode("ascii"))
        pdf.write(b"0000000000 65535 f \n")
        for idx in range(1, total_objs + 1):
            pdf.write(f"{offsets[idx]:010d} 00000 n \n".encode("ascii"))
        pdf.write(b"trailer\n")
        pdf.write(f"<< /Size {total_objs + 1} /Root 1 0 R >>\n".encode("ascii"))
        pdf.write(b"startxref\n")
        pdf.write(f"{xref_offset}\n".encode("ascii"))
        pdf.write(b"%%EOF\n")

        output_path.write_bytes(pdf.getvalue())

    def write_docx_from_text(output_path: Path, text: str) -> None:
        from xml.sax.saxutils import escape

        text = text or ""
        if not text.strip():
            text = "No se pudo extraer texto del documento."

        paragraphs = text.splitlines()

        def paragraph_xml(value: str) -> str:
            if not value.strip():
                return "<w:p/>"
            safe = escape(value)
            return (
                '<w:p><w:r><w:t xml:space="preserve">'
                f"{safe}"
                "</w:t></w:r></w:p>"
            )

        body = "".join(paragraph_xml(p) for p in paragraphs)
        content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
        rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>
"""
        document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("word/document.xml", document_xml)

    def write_doc_rtf(output_path: Path, text: str) -> None:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            text = "No se pudo extraer texto del documento."

        # Minimal RTF that Word can open.
        def escape_rtf(value: str) -> str:
            value = value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            value = value.replace("\n", "\\par\n")
            return value

        rtf = "{\\rtf1\\ansi\\deff0\n{\\fonttbl{\\f0 Times New Roman;}}\\f0\\fs24\n"
        rtf += escape_rtf(text)
        rtf += "\n}\n"
        # RTF header declares ANSI. Write as Windows-1252 so Word/libreoffice don't show mojibake (e.g., "número").
        output_path.write_bytes(rtf.encode("cp1252", errors="replace"))

    for fr in file_results:
        path = fr.get("path")
        filename = fr.get("filename") or "document"
        if not path:
            continue

        # Prefer extracted normalized text; fall back to QA/doc text if extraction failed.
        extraction_result = extraction_by_path.get(path)
        extracted = None
        normalized_text = ""
        if extraction_result and getattr(extraction_result, "success", False) and getattr(extraction_result, "extracted_data", None):
            extracted = extraction_result.extracted_data
            normalized_text = extracted.normalized_text or extracted.raw_text or ""
        if not normalized_text:
            normalized_text = fr.get("doc_text") or ""

        system_corrections = []
        if extracted and getattr(extracted, "additional_fields", None):
            system_corrections = extracted.additional_fields.get("system_corrections") or []

        # Best-effort text patching using applied system corrections.
        patched_text = normalized_text
        for entry in system_corrections:
            before = entry.get("before") or ""
            after = entry.get("after") or ""
            field = (entry.get("field") or "").lower()
            if not before or not after or before == after:
                continue
            if before in patched_text:
                patched_text = patched_text.replace(before, after)
            if field == "rut":
                before_digits = re.sub(r"[^0-9]", "", before)
                after_digits = re.sub(r"[^0-9]", "", after)
                if before_digits and after_digits and before_digits in patched_text:
                    patched_text = patched_text.replace(before_digits, after_digits)

        # Apply Smart Auto-Fix corrections
        smart_fix_corrections = results.get("smart_fix_corrections") or []
        current_filename = filename
        for correction in smart_fix_corrections:
            # Only apply corrections for this specific file
            if correction.get("file") != current_filename:
                continue

            old_val = correction.get("old") or ""
            new_val = correction.get("new") or ""
            field = (correction.get("field") or "").lower()

            if not old_val or not new_val or old_val == new_val:
                continue

            # Apply text replacement
            if old_val in patched_text:
                patched_text = patched_text.replace(old_val, new_val)

            # For date fields, also try to find and replace date formats
            if "date" in field:
                # Try to find variations of the date format
                from datetime import datetime
                try:
                    # Parse and try multiple formats
                    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                        try:
                            dt = datetime.strptime(old_val, fmt)
                            # Try to find this date in other formats in the text
                            for search_fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d de %B de %Y"]:
                                try:
                                    old_formatted = dt.strftime(search_fmt)
                                    new_dt = datetime.strptime(new_val, "%Y-%m-%d")
                                    new_formatted = new_dt.strftime(search_fmt)
                                    if old_formatted in patched_text:
                                        patched_text = patched_text.replace(old_formatted, new_formatted)
                                except:
                                    continue
                            break
                        except:
                            continue
                except:
                    pass

        # Ensure encoding/whitespace normalization on final output text (fixes BOM + mojibake like "número").
        patched_text = TextNormalizer.normalize_text(patched_text)

        validation = fr.get("validation") or {}
        validation_issues = validation.get("issues") or []
        expired_issue = next(
            (i for i in validation_issues if (i.get("issue_type") or "") == "expired_document"),
            None,
        )

        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem or "document")
        try:
            fixed_path = fixed_dir_root / Path(filename).name  # same name + same extension
            if fixed_path.exists():
                fixed_path = fixed_dir_root / f"{safe_stem}_{uuid4().hex[:8]}{Path(filename).suffix}"

            fmt = (fr.get("file_format") or Path(filename).suffix.lstrip(".")).lower()
            if fmt == "pdf":
                write_pdf_from_text(fixed_path, patched_text)
                mime = "application/pdf"
            elif fmt == "docx":
                write_docx_from_text(fixed_path, patched_text)
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif fmt == "doc":
                write_doc_rtf(fixed_path, patched_text)
                mime = "application/msword"
            else:
                # Fallback: keep extension but store plain text if format is unknown.
                fixed_path.write_text(patched_text, encoding="utf-8")
                mime = "text/plain"

            fr["fixed_outputs"] = [
                {"label": "Fixed document", "path": str(fixed_path), "mime": mime},
            ]
        except Exception as exc:
            fixed_dir_errors.append(f"{filename}: {exc}")

    if fixed_dir_errors:
        results["fixed_output_errors"] = fixed_dir_errors
    else:
        results.pop("fixed_output_errors", None)
    results["file_results"] = file_results
    return results


def collect_fixed_downloads(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Collect all fixed output downloads from results"""
    fixed_downloads: List[Dict[str, str]] = []
    for fr in results.get("file_results") or []:
        for item in fr.get("fixed_outputs") or []:
            fixed_downloads.append(
                {
                    "source": fr.get("filename") or "document",
                    "label": item.get("label") or "Download fixed output",
                    "path": item.get("path") or "",
                    "mime": item.get("mime") or "application/octet-stream",
                }
            )
    return [d for d in fixed_downloads if d.get("path")]
