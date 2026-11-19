# Matching baseline demo

Owner: Damir. This is an initial demonstration, not a validated recommender.

## Run the demo

Install `requirements.txt` in the project environment. Prepare the source
dataset using `scripts/prepare_data.py` as described in `data_audit.md`.
The notebook reads the generated `data/sample/sample_jobs.csv`.
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

Job skill labels are produced by the shared dictionary-based
`src.feature_extraction.extract_skills` function. The notebook no longer
uses the temporary demo vocabulary. Candidate skill aliases must follow
the same canonical naming conventions; coverage comparison itself only
strips whitespace and ignores case.

A score of 1 for skill coverage means all detected vocabulary labels match,
even if only one label was detected. Different score distributions and the
text-only fallback limit the interpretation of hybrid rankings. No claim
that hybrid improves quality is supported by this demo.

Historical 2023–2024 postings are not current openings. The sample is not a
market-representative analysis. Experience, mandatory eligibility, user
preferences, and human relevance metrics
remain future work. Scores are not interview or offer probabilities.

## Independent job-category ranking

`rank_job_categories(scored_jobs, job_categories, score_column="text_score",
m=5)` ranks job-function categories by the mean score of their top-m jobs.

Category links contain one job_id/category pair per row. A job may belong
to multiple categories. Duplicate links are removed; unknown scores are
excluded. Outputs report both jobs used and jobs available.

The user's selected category is compared with the completed ranking and
does not influence its calculation. Top-10 job recommendations remain
available separately.

The notebook demonstrates three synthetic profiles, alternative categories,
and sensitivity to text weights of 0.5, 0.75, and 1.0 and m values of 3, 5,
and 10. No configuration has been established as optimal. Top-m averaging
does not eliminate category-size effects.