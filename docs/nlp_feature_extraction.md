# JobScope — NLP & Skill Extraction Module

**Project:** JobScope — Job Market Intelligence & Personalized Career Recommendation System  
**Course:** NYU DS-GA 1007, Fall 2026  
**Owner:** Jessy (NLP, Feature Extraction & Evaluation)  
**Status:** MVP ready for team review  
**Target review:** November 12, 2026

## 1. Purpose and Scope

This module converts raw LinkedIn job descriptions into **deduplicated, standardized skill lists** for downstream candidate–job matching. It is a transparent, dictionary-based baseline intended to work across multiple occupations, not only data-science positions.

**Pipeline:** `Raw JD → clean_description() → regex / alias matching → canonical skill list`

The output can support Damir's skill-coverage component alongside TF-IDF cosine similarity. This module **does not** implement TF-IDF, ranking, candidate profiling, or the complete recommendation system.

## 2. Repository Files

| Path | Purpose |
| --- | --- |
| `src/feature_extraction.py` | Text normalization, dictionary loading, and skill extraction |
| `data/reference/skill_dictionary.json` | Curated canonical skills, categories, and aliases |
| `data/reference/jd_validation_sample.csv` | Ten cross-occupation job descriptions for a pilot evaluation |
| `data/reference/jd_skill_annotations.csv` | Reviewed pilot skill labels and annotation notes |
| `notebooks/02_feature_extraction.ipynb` | Data inspection, demonstrations, and pilot evaluation |
| `tests/test_feature_extraction.py` | Automated unit and import-contract tests |

The source job descriptions are from the LinkedIn Kaggle postings dataset (`data/raw/linkedin_kaggle/postings.csv`). Its relevant fields are `job_id`, `title`, and `description`. Large raw data files are not required for the small standalone extraction demo if the reference samples are available.

**Important distinction:** The dataset's `mappings/skills.csv` and `jobs/job_skills.csv` contain LinkedIn category codes and job–category relationships. They are **not** the canonical skill dictionary used by this extractor.

## 3. Extraction Method

### 3.1 Text preprocessing

`clean_description(text)` decodes HTML entities, removes HTML tags, normalizes Unicode and common punctuation, collapses repeated whitespace, and handles missing or non-string inputs by returning an empty string. It intentionally preserves skill-bearing symbols, such as `C++`, `C#`, `.NET`, and `R`.

### 3.2 Curated skill dictionary

`skill_dictionary.json` (version `0.1.0`) contains **95 canonical skills**, **105 additional aliases**, and six occupational categories:

| Category | Canonical skills |
| --- | ---: |
| Data & Technology | 20 |
| Finance & Accounting | 15 |
| Marketing & Sales | 15 |
| Business & Management | 15 |
| Engineering | 15 |
| Healthcare & Administration | 15 |
| **Total** | **95** |

Each record includes a `name`, a `category`, and an `aliases` list. For example, `MS Excel` and `Microsoft Excel` normalize to `Excel`. Categories are metadata for organizing the dictionary, **not** skills automatically assigned to a posting.

### 3.3 Pattern matching and normalization

`extract_skills(description, dictionary_path=None)`:

1. Cleans the input using `clean_description()`.
2. Loads and compiles case-aware regular-expression patterns for canonical names and aliases (cached by dictionary path).
3. Applies boundaries to reduce unintended substring matches, while supporting names containing punctuation.
4. Adds a contextual safeguard for the single-letter `R` language name.
5. Deduplicates by canonical name and returns skills in first-occurrence order.

This is **keyword/alias recognition**, not contextual named-entity recognition or an LLM-based extractor. Consequently, variations not explicitly listed in the dictionary may be missed.

## 4. Quick Start and Damir Integration

Run the following from the repository root (`Py3onistas/`):

```python
from src.feature_extraction import clean_description, extract_skills

job_description = (
    "We are looking for an analyst proficient in "
    "Python, SQL, MS Excel, and Tableau."
)

cleaned_text = clean_description(job_description)
job_skills = extract_skills(job_description)

print("Cleaned description:", cleaned_text)
print("Extracted skills:", job_skills)
# Expected: ['Python', 'SQL', 'Excel', 'Tableau']
```

**Interface contract for Damir:**

- **Input:** Raw job-description string (`None` and empty input return an empty list in the current implementation).
- **Output:** `list` of unique canonical skill-name strings, e.g., `['Python', 'SQL']`.
- **Ordering:** First appearance in the description; do not interpret list order as importance or proficiency.
- **Dependency:** The default dictionary is read relative to the repository's `src/` module, so calls from the repository root do not require a hard-coded laptop-specific path.
- **Optional override:** Supply `dictionary_path` to use a different dictionary JSON file.
- **Integration suggestion:** Compare canonical names against a candidate skill profile using set overlap/skill coverage; combine that score with Damir's separate TF-IDF cosine similarity. Agree on how empty skill lists and score weights are handled before integration.

The current import-contract test confirms that the public function is callable. It does **not** establish that the end-to-end ranking system has been integrated or validated.

## 5. Reproducibility and Tests

From the repository root:

```bash
python -m pytest tests/test_feature_extraction.py -v
```

On the recorded local run (Python 3.12.2, pytest 7.4.4), **20 tests passed**. Coverage includes preprocessing, empty inputs, canonical names and aliases, duplicate removal, symbol-preserving matches, word-boundary behavior, `R` context safeguards, cross-occupation examples, a custom dictionary, and Damir's import contract.

To reproduce the notebook, open `notebooks/02_feature_extraction.ipynb` using a Python environment with the repository dependencies, make sure the referenced CSV files are present, and run the cells from top to bottom. The notebook was also used for data inspection and manual-label evaluation.

## 6. Pilot Evaluation

A **reviewed pilot annotation set of ten job descriptions** was selected from different occupational areas. Evaluation compares predicted canonical skill sets with reviewed `expected_skills` labels **within the current 95-skill vocabulary**.

| Metric | Pilot result |
| --- | ---: |
| True positives (TP) | 20 |
| False positives (FP) | 2 |
| False negatives (FN) | 17 |
| Micro precision | 0.909 |
| Micro recall | 0.541 |
| Micro F1 | 0.678 |

Definitions: `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`, and `F1 = 2PR / (P + R)`.

**Interpretation:** The baseline is relatively precise on this small set but misses many labeled skills. This supports prioritizing recall improvements, especially alternate phrasing and coverage outside data/technology roles.

**Evaluation caveats:** Ten postings are not representative of the US labor market. The labels were reviewed after an AI-assisted draft rather than constructed as a fully independent benchmark. Results should be described as **pilot evaluation**, not as population-level extraction accuracy or recommendation `Precision@K`. Skills outside the 95-item lexicon are separate dictionary-coverage gaps, not captured by this within-vocabulary metric.

## 7. Known Limitations and Follow-up Work

- **Dictionary coverage:** The initial 95 skills cannot cover the range of occupations and skills in the US job market. Maintain a separate log of out-of-vocabulary skills encountered during validation.
- **Recall and wording:** Phrases such as `administer medications` or broad Office-suite references may not match current canonical terms and aliases. Review the actual JD text before adding aliases.
- **Ambiguous skills:** Very short abbreviations (`R`, `ML`, `JS`) and broad words (`Manufacturing`, `Forecasting`) can produce context-dependent false positives. Regex safeguards reduce, but cannot eliminate, these errors.
- **Semantics:** The extractor does not distinguish required from preferred skills, proficiency levels, negation, or past experience. A mention is not proof of candidate competence or job importance.
- **Evaluation size:** Increase the number and diversity of independently annotated descriptions before treating metrics as reliable.
- **Downstream integration:** Damir should confirm canonical candidate-skill formatting, missing-skill behavior, hybrid weighting, and ranking evaluation in the recommendation pipeline.

## 8. Meeting Demo (Approximately 3 Minutes)

1. **Objective (30 seconds):** Explain the raw-JD-to-standardized-skills interface and its cross-occupation scope.
2. **Method (45 seconds):** Show preprocessing, the six-category dictionary, regex boundaries, alias normalization, and deduplication.
3. **Live demo (45 seconds):** Call `extract_skills()` on representative Data, Finance, and Healthcare descriptions; show canonical outputs.
4. **Pilot results (30 seconds):** Report precision `0.909`, recall `0.541`, and F1 `0.678` with the small-sample caveat; discuss false negatives.
5. **Integration (30 seconds):** Show the single import line used by Damir and the passing unit tests; distinguish interface readiness from complete ranking integration.

## 9. Review Checklist

- [ ] Aileen confirms JD inputs, data field names, and sample provenance.
- [ ] Damir confirms the function signature and canonical skill-name format are suitable for skill-coverage scoring.
- [ ] Teammates reproduce the sample extraction and run the pytest suite.
- [ ] The team agrees on vocabulary extensions and annotation protocol for the next iteration.
- [ ] The feature-extraction PR is reviewed before merging into `main`.
