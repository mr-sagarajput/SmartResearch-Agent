"""
Professional PDF Report Builder with Full Unicode Normalization.
================================================================
Eliminates black boxes (■) by sanitizing non-ASCII hyphens, dashes, and smart quotes
into standard ReportLab-compatible characters.
"""

from datetime import datetime
import html
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from core.config import OUTPUT_DIR


def sanitize_text_for_pdf(text: str) -> str:
    """
    Normalizes fancy web unicode characters that cause black box (■) artifacts:
    - En-dash, Em-dash, minus, non-breaking hyphens -> standard ASCII '-'
    - Smart quotes / curly quotes -> standard ASCII quotes
    - Bullets, ellipsis, non-breaking spaces -> clean safe equivalents
    """
    if not text:
        return ""

    # Replace various Unicode hyphens, dashes, quotes, and math symbols
    replacements = {
        "\u2010": "-",  # Hyphen
        "\u2011": "-",  # Non-breaking hyphen
        "\u2012": "-",  # Figure dash
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2015": "-",  # Horizontal bar
        "\u2212": "-",  # Minus sign
        "\u00ad": "-",  # Soft hyphen
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201a": "'",  # Single low-9 quote
        "\u201b": "'",  # Single high-reversed-9 quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u201e": '"',  # Double low-9 quote
        "\u2026": "...", # Ellipsis
        "\u00a0": " ",  # Non-breaking space
        "\u2002": " ",  # En space
        "\u2003": " ",  # Em space
        "\u2009": " ",  # Thin space
        "\u2022": "•",  # Standard bullet
        "\u00b1": "+/-", # Plus-minus sign
        "\u00d7": "*",   # Multiplication sign
        "\u00f7": "/",   # Division sign
        "\u2260": "!=",  # Not equal to
        "\u2264": "<=",  # Less-than or equal to
        "\u2265": ">=",  # Greater-than or equal to
        "\u2248": "~",   # Almost equal to
        "\u221e": "inf", # Infinity
    }

    for char, safe_replacement in replacements.items():
        text = text.replace(char, safe_replacement)

    # Encode to latin-1 safe representation if any obscure characters remain
    safe_chars = []
    for c in text:
        if ord(c) < 128 or c in ("•",):
            safe_chars.append(c)
        else:
            # Fallback for unexpected unicode symbols
            safe_chars.append(" ")

    cleaned = "".join(safe_chars)
    # Escape HTML special chars so ReportLab Platypus doesn't choke on '<' or '&' (keep quotes unescaped for clean PDF text)
    return html.escape(cleaned, quote=False).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")


def get_unique_filename(base_name: str, ext: str = ".pdf") -> Path:
    clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)[:50]
    date_str = datetime.now().strftime("%Y-%m-%d")
    candidate = OUTPUT_DIR / f"research_{clean_name}_{date_str}{ext}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = OUTPUT_DIR / f"research_{clean_name}_{date_str}_{counter}{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def generate_research_pdf_report(
    topic: str,
    summary_text: str,
    sources_data: List[Dict[str, Any]],
) -> Path:
    """
    Builds a professional PDF research report and saves it to output/.
    Returns the absolute Path of the created file.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_path = get_unique_filename(topic, ext=".pdf")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        alignment=0,
    )

    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#64748b"),
    )

    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    story = []

    # Title & Metadata
    safe_topic = sanitize_text_for_pdf(topic)
    story.append(Paragraph("Autonomous Research Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Topic: <b>{safe_topic}</b>", ParagraphStyle("Sub", fontName="Helvetica", fontSize=11, leading=14)))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y - %H:%M')} | Engine: SmartResearch AI", meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # Executive Summary (Cleaned & Sanitized)
    story.append(Paragraph("Executive Summary", h2_style))
    for para in summary_text.split("\n\n"):
        clean_para = sanitize_text_for_pdf(para.strip())
        if clean_para:
            story.append(Paragraph(clean_para, body_style))
    story.append(Spacer(1, 10))

    # Key Source Findings
    story.append(Paragraph(f"Detailed Source Findings ({len(sources_data)} Sources)", h2_style))
    for idx, src in enumerate(sources_data, 1):
        clean_title = sanitize_text_for_pdf(src.get("title", f"Source {idx}"))
        raw_url = src.get("url", "")
        clean_notes = sanitize_text_for_pdf(src.get("notes", ""))

        header_text = f"<b>{idx}. {clean_title}</b><br/><font color='#2563eb' size='8'>{raw_url}</font>"
        story.append(Paragraph(header_text, body_style))

        for line in clean_notes.splitlines():
            line = line.strip()
            if line:
                if not line.startswith("•") and not line.startswith("-"):
                    line = f"• {line}"
                story.append(Paragraph(line, bullet_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    return pdf_path
