"""Unit and integration tests for JobScope skill extraction."""

import json
import sys
from pathlib import Path

import pytest

# Support running pytest from the repository root or from the tests directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import clean_description, extract_skills


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        (None, ""),
        ("   ", ""),
        ("<p>Python &amp; SQL</p>", "Python & SQL"),
        ("Python  \n SQL\tExcel", "Python SQL Excel"),
        ("Experience with Python—SQL and “Excel”.", 'Experience with Python-SQL and "Excel".'),
        ("C++, C#, .NET, R, Node.js", "C++, C#, .NET, R, Node.js"),
    ],
)
def test_clean_description(raw, expected):
    assert clean_description(raw) == expected


def test_empty_description_returns_no_skills():
    assert extract_skills("") == []
    assert extract_skills(None) == []
    assert extract_skills("   ") == []


def test_canonical_skills_and_aliases():
    result = extract_skills("Python, SQL, MS Excel, and Tableau.")
    assert result == ["Python", "SQL", "Excel", "Tableau"]


def test_deduplication_and_alias_normalization():
    result = extract_skills("Python, Python, MS Excel, Microsoft Excel, SQL, SQL.")
    assert result == ["Python", "Excel", "SQL"]


def test_special_symbol_skills():
    result = extract_skills("Experience with C++, C#, and .NET.")
    assert {"C++", "C#", ".NET"}.issubset(set(result))


def test_word_boundaries():
    result = extract_skills("We use JavaScript and AutoCAD.")
    assert "JavaScript" in result
    assert "Java" not in result
    assert "AutoCAD" in result
    assert "CAD" not in result


def test_single_letter_r_requires_context():
    assert "R" not in extract_skills("The bar is high and our team runs events.")
    assert "R" in extract_skills("Experience in Python, R, and SQL.")


@pytest.mark.parametrize(
    "description, expected",
    [
        ("Financial Modeling, QuickBooks, and GAAP", {"Financial Modeling", "QuickBooks", "GAAP"}),
        ("SEO, Google Analytics, and Salesforce", {"SEO", "Google Analytics", "Salesforce"}),
        ("AutoCAD, SolidWorks, and MATLAB", {"AutoCAD", "SolidWorks", "MATLAB"}),
        ("HIPAA, EMR, and CPR", {"HIPAA Compliance", "Electronic Medical Records", "CPR"}),
        ("Project Management and Agile methodology", {"Project Management", "Agile"}),
    ],
)
def test_cross_occupation_matching(description, expected):
    assert expected.issubset(set(extract_skills(description)))


def test_custom_dictionary_path(tmp_path):
    dictionary = {
        "skills": [
            {"name": "ExampleSkill", "category": "Testing", "aliases": ["example alias"]}
        ]
    }
    dictionary_path = tmp_path / "skills.json"
    dictionary_path.write_text(json.dumps(dictionary), encoding="utf-8")

    assert extract_skills("The example alias is required.", dictionary_path) == ["ExampleSkill"]


def test_damir_import_contract():
    # Damir should be able to call this function without a notebook or a custom path.
    from src.feature_extraction import extract_skills as recommender_extractor

    result = recommender_extractor("Python and SQL required")
    assert isinstance(result, list)
    assert {"Python", "SQL"}.issubset(set(result))
    assert all(isinstance(skill, str) for skill in result)
