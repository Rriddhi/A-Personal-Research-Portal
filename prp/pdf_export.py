"""
Phase 3: PDF export for evidence table, synthesis memo, and full run.
Uses reportlab. Returns bytes for Streamlit download_button.
"""
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def _add_page_number(canvas, doc):
    """Footer with page number."""
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(
        letter[0] / 2.0,
        0.5 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def evidence_table_to_pdf(
    evidence_rows: list[dict],
    title: str,
    subtitle: str | None = None,
) -> bytes:
    """
    Return PDF bytes for evidence table rows.
    Columns: claim, evidence, citation, source, page.
    Row dict keys can be: Claim, Evidence snippet, Citation, Confidence, Notes (artifacts style);
    source/page default to — if missing.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=14,
        spaceAfter=6,
    )
    story.append(Paragraph(title, title_style))
    if subtitle:
        story.append(Paragraph(subtitle, styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    if not evidence_rows:
        story.append(Paragraph("No evidence rows.", styles["Normal"]))
        doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
        return buf.getvalue()

    # Map artifact-style keys to display columns; # = row number
    col_claim = "Claim"
    col_evidence = "Evidence snippet"
    col_citation = "Citation"
    col_source = "Source"
    col_page = "Page"

    # Align with artifacts display limits (claim_len=1200, snippet_len=1000)
    claim_len, snippet_len = 1200, 1000
    data = [["#", col_claim, col_evidence, col_citation, col_source, col_page]]
    for i, r in enumerate(evidence_rows, 1):
        claim = (r.get("Claim") or r.get("claim") or "")[:claim_len]
        evidence = (r.get("Evidence snippet") or r.get("evidence") or "")[:snippet_len]
        citation = (r.get("Citation") or r.get("citation") or "")[:300]
        source = (r.get("Source") or r.get("source") or citation.split("·")[0].strip() if citation else "—")[:200]
        page = str(r.get("page") or r.get("Page") or "—")[:20]
        data.append([str(i), claim, evidence, citation, source, page])

    # Total ~7 inch (letter 8.5 - margins); # column narrow
    col_widths = [0.35 * inch, 1.25 * inch, 2.0 * inch, 1.6 * inch, 1.2 * inch, 0.5 * inch]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
            ]
        )
    )
    story.append(t)
    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buf.getvalue()


def memo_to_pdf(
    memo_md: str,
    title: str,
    metadata: dict | None = None,
) -> bytes:
    """
    Render a synthesis memo (markdown-ish) into a readable PDF.
    Simple headings (##) and paragraphs; no full markdown parsing.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Heading1"]))
    if metadata:
        meta_str = " · ".join(f"{k}: {v}" for k, v in (metadata or {}).items() if v)
        if meta_str:
            story.append(Paragraph(meta_str, styles["Normal"]))
            story.append(Spacer(1, 0.15 * inch))

    # Simple split: ## as heading, rest as paragraph (replace newlines with <br/> for Paragraph)
    current_para: list[str] = []
    for line in (memo_md or "").splitlines():
        line = line.strip()
        if not line:
            if current_para:
                text = " ".join(current_para).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(text, styles["Normal"]))
                story.append(Spacer(1, 0.1 * inch))
                current_para = []
            continue
        if line.startswith("##"):
            if current_para:
                text = " ".join(current_para).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(text, styles["Normal"]))
                story.append(Spacer(1, 0.1 * inch))
                current_para = []
            heading = line.lstrip("#").strip()
            story.append(Paragraph(heading, styles["Heading2"]))
            story.append(Spacer(1, 0.08 * inch))
        else:
            current_para.append(line)
    if current_para:
        text = " ".join(current_para).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(text, styles["Normal"]))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buf.getvalue()


def run_to_pdf(run: dict, title: str) -> bytes:
    """
    PDF containing: question, answer, citations list, retrieved evidence chunks (top-k), and timestamp.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Heading1"]))
    ts = run.get("timestamp") or run.get("metadata", {}).get("timestamp") or ""
    if ts:
        story.append(Paragraph(f"Timestamp: {ts}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    question = run.get("query_text") or run.get("query") or ""
    story.append(Paragraph("Question", styles["Heading2"]))
    story.append(Paragraph(question.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Answer", styles["Heading2"]))
    answer = (run.get("answer") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # One paragraph; long text will wrap
    story.append(Paragraph(answer, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Citations", styles["Heading2"]))
    mapping = run.get("citation_mapping") or []
    if mapping:
        for m in mapping[:30]:
            apa = m.get("apa") or ""
            cid = m.get("chunk_id") or ""
            story.append(Paragraph(f"• {apa} → {cid}", styles["Normal"]))
    else:
        story.append(Paragraph("No citations.", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Retrieved evidence (top chunks)", styles["Heading2"]))
    chunks = run.get("retrieved_chunks") or []
    for i, c in enumerate(chunks[:15], 1):
        cid = c.get("chunk_id") or ""
        sid = c.get("source_id") or ""
        text = (c.get("text") or c.get("text_preview") or "").strip()
        if not text:
            text = "(chunk text not stored)"
        text = text[:1200].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(f"<b>{i}. [{sid}] {cid}</b>", styles["Normal"]))
        story.append(Paragraph(text, styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buf.getvalue()
