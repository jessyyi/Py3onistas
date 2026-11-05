import html
import re
import unicodedata


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