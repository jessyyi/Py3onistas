# Matching baseline demo

Owner: Damir. This is an initial demonstration, not a validated recommender.

## Run the demo

Install `requirements.txt` in the project environment. Prepare the source
dataset using `scripts/prepare_data.py` as described in `data_audit.md`.
The notebook reads the generated `data/processed/sample_jobs.csv`.
Raw and processed datasets are local, ignored files.

Open `notebooks/03_recommendation.ipynb`, select the project environment,
and restart the kernel and run all cells. The notebook assumes its working
directory is `notebooks/`. It demonstrates three synthetic profiles on
a 500-posting sample, with Top-10 text, skill, and hybrid rankings.

Run checks from the repository root with `python -m pytest tests -q`.

## Public interface

```python
from src.matching import TextMatcher
from src.recommendation import recommend_top_k

matcher = TextMatcher().fit(jobs)
top_jobs = recommend_top_k(candidate, jobs, k=10, matcher=matcher)

# skill_labels contains lists of normalized skill names, prepared upstream.
top_hybrid = recommend_top_k(
    candidate, jobs, k=10, matcher=matcher,
    method="hybrid", skills_column="skill_labels", text_weight=0.5,
)
```

Add skill labels before fitting the matcher. The supplied jobs table must
equal the fitted table, including row order. Reuse the fitted matcher across
candidate requests. `matcher` is an explicit addition to the initial shared
API to avoid fitting TF-IDF on every recommendation request.

Candidate dictionaries contain `candidate_id`, `skills` (list of strings),
`education` (string), and `experience` (string). Candidate IDs label outputs;
they are excluded from matching text. Preferences are not implemented yet.

## Scoring and explanations

Text scoring fits English-stop-word TF-IDF on descriptions only and compares
each candidate with jobs using cosine similarity. It does not build a dense
job-to-job similarity matrix.

Skill coverage is the number of matched distinct normalized labels divided
by the number of distinct detected job labels. Label comparison strips
whitespace and ignores case; aliases must be normalized upstream.
Unmatched job labels are reported as unverified, not proven absent skills.
Empty job label lists produce an unknown skill score. Skill-only ranking
excludes those rows and can return fewer than K results.

Hybrid scoring uses `text_weight * text_score + (1 - text_weight) * skill_score`
when skill evidence is available, and text score alone otherwise.
`hybrid_uses_skills` identifies that distinction. The demo uses equal weights;
these are not calibrated or optimized. All methods break ties by job ID.

## Limits and next integration steps

The notebook's `extract_demo_skills` is a temporary vocabulary-based mention
detector for three demonstration domains. It is not Jessy's completed
extractor and does not interpret mandatory versus optional requirements.
Replace its output with normalized extraction results when available.

A score of 1 for skill coverage means all detected vocabulary labels match,
even if only one label was detected. Different score distributions and the
text-only fallback limit the interpretation of hybrid rankings. No claim
that hybrid improves quality is supported by this demo.

Historical 2023–2024 postings are not current openings. The sample is not a
market-representative analysis. Experience, mandatory eligibility, user
preferences, independent category rankings, and human relevance metrics
remain future work. Scores are not interview or offer probabilities.
