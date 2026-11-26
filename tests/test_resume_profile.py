from src.resume_profile import extract_profile_evidence, parse_sections
from src.resume_studio import build_resume_docx
from io import BytesIO
from docx import Document


def test_explicit_sections_are_preserved():
    text = "Education\nNYU, MS Data Science\nExperience\nData Analyst Intern, Example Co\nSkills\nPython, SQL"
    result = extract_profile_evidence(text)
    assert "NYU" in result["education_summary"]
    assert "Example Co" in result["experience_summary"]
    assert result["education_found"] and result["experience_found"]


def test_fallback_does_not_fabricate_history():
    result = extract_profile_evidence("A student interested in data visualization.")
    assert not result["education_found"]
    assert not result["experience_found"]
    assert "confirm manually" in result["education_summary"]


def test_resume_export_round_trip():
    data = build_resume_docx({"name": "Example User", "education": "MS Data Science", "skills": "Python, SQL"})
    doc = Document(BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Example User" in text
    assert "MS Data Science" in text
    assert "Python, SQL" in text
