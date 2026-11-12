import html
import re
import unicodedata
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

def clean_description(text: str) -> str:
    """
    Clean and normalize a job description.

    Handles HTML, Unicode punctuation, and whitespace while
    preserving skill-specific symbols such as C++, C#, and .NET.

    Args:
        text: Raw job description.

    Returns:
        Cleaned job description string.
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Replace HTML tags with spaces
    text = re.sub(r"<[^>]*>", " ", text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Normalize common punctuation
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# Locate the default skill dictionary
DEFAULT_DICTIONARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "skill_dictionary.json"
)


@lru_cache(maxsize=8)
def _load_patterns(dictionary_path: str) -> tuple:
    """
    Load the skill dictionary and compile regex patterns.

    Args:
        dictionary_path: Absolute path to the skill dictionary.

    Returns:
        Compiled patterns mapped to canonical skill names.
    """
    with open(dictionary_path, "r", encoding="utf-8") as file:
        dictionary = json.load(file)

    patterns = []

    for skill in dictionary["skills"]:
        canonical_name = skill["name"]

        # Match both canonical names and aliases
        terms = [canonical_name] + skill.get("aliases", [])

        for term in terms:
            if not isinstance(term, str) or not term.strip():
                continue

            escaped_term = re.escape(term.strip())

            # Prevent partial matches while preserving skill symbols
            regex = rf"(?<![\w.+#]){escaped_term}(?![\w+#])"

            # Short abbreviations require exact capitalization
            flags = re.IGNORECASE

            if term in {"R", "ML", "DL", "JS", "AP", "AR", "GL", "QA"}:
                flags = 0

            patterns.append(
                (canonical_name, re.compile(regex, flags))
            )

    return tuple(patterns)


def extract_skills(
    description: str,
    dictionary_path: Optional[Union[str, Path]] = None
) -> list:
    """
    Extract standardized skills from a job description.

    Uses regex matching, aliases, and deduplication.

    Args:
        description: Raw job description text.
        dictionary_path: Optional custom skill dictionary path.

    Returns:
        A deduplicated list of canonical skill names.
    """
    # Preprocess the raw job description
    text = clean_description(description)

    if not text:
        return []

    # Resolve the dictionary path
    path = (
        Path(dictionary_path)
        if dictionary_path is not None
        else DEFAULT_DICTIONARY_PATH
    )

    candidates = []

    # Match canonical skill names and aliases
    for name, pattern in _load_patterns(str(path.resolve())):

        for match in pattern.finditer(text):

            # Reduce false positives for the R programming language
            if name == "R" and match.group() == "R":
                context = text[
                    max(0, match.start() - 45):
                    match.end() + 45
                ]

                if not re.search(
                    r"\b(python|sql|programming|language|statistical|"
                    r"statistics|analytics|tableau|sas)\b",
                    context,
                    flags=re.IGNORECASE
                ):
                    continue

            candidates.append(
                (match.start(), -len(match.group()), name)
            )

    # Sort skills by first occurrence in the description
    candidates.sort()

    extracted_skills = []
    seen = set()

    # Remove duplicate skills and normalize aliases
    for _, _, name in candidates:
        if name not in seen:
            seen.add(name)
            extracted_skills.append(name)

    return extracted_skills