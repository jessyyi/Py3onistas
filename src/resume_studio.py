"""Editable, private in-session resume templates with DOCX export."""

from __future__ import annotations

from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt, RGBColor

TEMPLATES = {
    "Technical Classic": {"color": (33, 49, 70), "font": "Aptos", "size": 10},
    "Editorial Modern": {"color": (81, 56, 69), "font": "Aptos", "size": 10},
    "Research Minimal": {"color": (47, 82, 72), "font": "Aptos", "size": 10},
}


def build_resume_docx(profile: dict[str, str], template: str = "Technical Classic") -> bytes:
    """Generate a simple single-column resume using only user-confirmed content."""
    if template not in TEMPLATES:
        raise ValueError("Unknown resume template")
    settings = TEMPLATES[template]
    document = Document()
    sec = document.sections[0]
    sec.top_margin = Inches(0.65)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = settings["font"]
    normal.font.size = Pt(settings["size"])
    normal.paragraph_format.space_after = Pt(3)

    def add_header(title: str) -> None:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(10)
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(title.upper())
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(*settings["color"])

    name = (profile.get("name") or "Your Name").strip()
    name_p = document.add_paragraph()
    name_p.alignment = 1
    r = name_p.add_run(name)
    r.bold = True
    r.font.size = Pt(18 if template != "Editorial Modern" else 20)
    r.font.color.rgb = RGBColor(*settings["color"])
    contact = " | ".join(filter(None, [profile.get("email", "").strip(), profile.get("location", "").strip(), profile.get("links", "").strip()]))
    if contact:
        p = document.add_paragraph(contact)
        p.alignment = 1
    for heading, field in [
        ("Professional Summary", "summary"),
        ("Education", "education"),
        ("Experience", "experience"),
        ("Projects", "projects"),
        ("Skills", "skills"),
    ]:
        value = profile.get(field, "").strip()
        if not value:
            continue
        add_header(heading)
        for line in value.splitlines():
            cleaned = line.strip().lstrip("•*- ").strip()
            if not cleaned:
                continue
            if heading in {"Experience", "Projects"} and line.lstrip().startswith(("•", "-", "*")):
                document.add_paragraph(cleaned, style="List Bullet")
            else:
                document.add_paragraph(cleaned)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
