# JobScope — Shared Module Interfaces

This document defines the initial integration contract for the three workstreams. Agree on any schema changes in a GitHub Issue before modifying downstream assumptions.

## Aileen — Data Engineering

Output: `clean_jobs: pandas.DataFrame`.

Required columns:
- `job_id`: unique posting identifier
- `title`: job title
- `description`: original or cleaned job-description text
- `location`: standardized U.S. location
- `job_function`: job-function tag(s), not industry
- `industry`: company industry classification(s), not job function

Optional fields: company name, compensation ranges, salary period, work arrangement, qualifications and experience. Multi-label categories should be represented consistently; decide exact representation together.

Join one-to-many tables carefully and document duplicate, missingness and salary-period handling. Use a reproducible sample for matching, and do not present balanced-sample proportions as U.S. market-wide shares.

## Jessy — NLP and Evaluation

Initial public functions in `src/feature_extraction.py`:

```python
def clean_description(text: str) -> str:
    ...

def extract_skills(description: str) -> list[str]:
    ...
```

Return a normalized list of selected skills using a limited cross-occupation vocabulary. Missing wording does not prove ineligibility. Document pattern matches, false positives and unverified requirements.

Evaluation uses independent 0/1/2 relevance ratings by two team members without exposing system scores to raters.

## Damir — Matching and Recommendation

Initial API:

```python
def recommend_top_k(candidate, jobs, k: int = 10):
    ...
```

Candidate profiles include declared skills, qualifications, experience and preferences. Compare skill coverage, TF-IDF cosine similarity and interpretable hybrid ranking; assess explicitly mandatory qualifications separately and mark unclear eligibility as unknown.

Career-category rankings must be generated independently of the category selected by a user, before comparing the two. Avoid favoring categories solely because they have more job postings.

## Integration and Git Workflow

- `main`: shared stable code
- `aileen-data`: data processing and EDA
- `jessy-nlp`: text features and evaluation
- `damir-matching`: matching, recommendation and optional app

Use Pull Requests to merge tested changes into `main`. Keep raw Kaggle CSVs in ignored `data/raw/` folders, not the repository. All paths should be relative to the project root.

## Upcoming milestones

- Nov 12, 2026: clean dataset/sample, skill extraction demo, and Top-K matching baseline
- Nov 19, 2026: market insights and independent career-category ranking
- Nov 26, 2026: evaluation results and feature freeze
- Dec 1, 2026: final code and presentation draft
- Dec 3, 2026: team presentation
