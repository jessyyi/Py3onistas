# JobScope: agreed project plan

**Team:** Py3onistas — Aileen, Jessy, Damir  
**Course:** NYU DS-GA 1007 (Fall 2026)  
**Status:** Proposal prepared. Full postings validation, EDA, and evaluation are in progress; no model-performance claims yet.

## Research scope

General **U.S.** job market using the historical LinkedIn Job Postings (2023–2024) dataset. These historical vacancies must not be shown as currently open.

- **Occupation / career category:** job-function tags from `job_skills.csv` + `skills.csv` (not fine-grained technical skills).
- **Industry:** company-industry mapping from `company_industries.csv` + `industries.csv`, with job industry tags as a separate dimension if needed.
- Multi-label dimensions require one-to-many-aware aggregations; avoid inflated counts after joins.

## Research questions

1. **Market intelligence:** How do skill requirements, pay, and job characteristics differ by occupation, industry, and location?
2. **Candidate–job matching:** Can a candidate profile and JD be matched with interpretable skill coverage and TF-IDF cosine similarity?
3. **Personalized recommendation:** Can we rank Top-K jobs and suggest career categories independently of the user's chosen category, then compare the two?

## Candidate data

Start with synthetic candidate profiles. Use real volunteers' information only with consent and appropriate minimization. Do not commit identifiable resumes or personal records.

## Architecture

Source CSVs → data validation / cleaning → job text and skills → two branches:
- Market EDA: occupation-skill heatmap, salary charts, job category patterns.
- Content-based matching: skill overlap + TF-IDF, separate eligibility review, user preference ranking → Top-K jobs, matched and unverified skills, category comparison.

Matching scores are **relative relevance**, not probability of interviews or offers. No arbitrary 50% relevance cutoff should be represented as validated.

## Evaluation

- Check key uniqueness, missingness, salary period, and text extraction examples.
- Independently label candidate-job relevance examples with two raters using a shared rubric, and report agreement.
- Compare skill-only, text-only, and hybrid ranking with Precision@K / NDCG@K when data permit.
- Keep train/test/evaluation candidate profiles separate if learning weights.

## Project ownership

- **Aileen:** data cleaning, integration, EDA.
- **Jessy:** feature extraction, evaluation.
- **Damir:** matching, recommendation, integration.

Shared: final slides and review. Streamlit is optional. Resume rewriting and recruiter JD generation are outside the current proposal.

## Key next action

Place the original Kaggle CSVs under `data/raw/` on each contributor's machine. Then run `scripts/prepare_data.py`, validate output counts and schemas, and start EDA / extraction tests.
