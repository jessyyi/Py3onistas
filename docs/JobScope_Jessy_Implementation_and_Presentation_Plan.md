# JobScope — Jessy's Evaluation, Product Implementation, and Presentation Plan

**Scope:** Evaluation figures before the next meeting; product implementation during the agreed Thanksgiving build window.  
**Project:** NYU DS-GA 1007 — JobScope / Py3onistas  
**Dataset:** Historical LinkedIn postings (2023–2024), not active job openings.

## 1. Completed Work and Why It Matters

- **NLP preprocessing and skill extraction:** Standardized job-description text and extracted dictionary-based skills. This produces interpretable job skill labels, enabling explanations beyond opaque TF-IDF scores. Existing coverage measured at 61.2% (306/500 jobs with at least one detected skill); this is detection coverage, **not extraction accuracy or recall**.
- **Integration with matching:** Provided normalized job skills for Damir's text, skill-only, and hybrid ranking pipeline; candidate-declared skills need consistent aliases.
- **Human evaluation design and analysis:** Constructed a 45-pair, three-profile, model-informed evaluation pool; collected Aileen and Damir ratings, plus Jessy's human-reviewed, AI-assisted judgments; calculated agreement and preliminary NDCG/Precision measures.
- **Error analysis:** Revealed that high detected-skill coverage does not establish overall job suitability, seniority fit, or professional eligibility. This directly motivates transparent ranking explanations and qualification caveats in the product.

## 2. Immediate Analysis Deliverables (Before Meeting)

**Code to add:** `scripts/analyze_evaluation_results.py` (figure-generation only). Do not modify `src/` core code.

**Inputs:**
- `data/processed/evaluation_results/model_comparison_by_profile.csv`
- `data/processed/evaluation_results/provisional_consensus.csv`
- `data/processed/evaluation_results/ranked_judged_jobs.csv`

**Run:** `python -m scripts.analyze_evaluation_results`

**Outputs:**
- `figures/evaluation/model_performance_comparison.png`
- `figures/evaluation/annotation_agreement.png`
- `figures/evaluation/ranking_error_analysis.png`
- `data/processed/evaluation_results/ranking_error_cases.csv`

**Presentation findings:**
- TF-IDF vs Hybrid NDCG@5: Data Science 0.798 vs 0.596; Finance 0.516 vs 0.650; Healthcare 0.463 vs 0.670.
- Skill-only is incomplete for two profiles. Do not treat N/A as zero.
- Aileen–Damir exact agreement 82.2%, quadratic-weighted kappa 0.816; other comparisons include AI-assisted labels.
- Key cases: Senior Data Analyst demoted from #3 to #8 for DS; Database Administrator promoted from #10 to #5 for Finance; Nurse Practitioner #2 in Healthcare Hybrid despite relevance 0.

## 3. Suggested Final Report Section (English Copy-Ready Draft)

### Human Relevance Evaluation and Explainability

We evaluated JobScope's ranking quality using three synthetic candidate profiles—Data Science, Finance, and Healthcare—and 15 judged jobs per profile. The 45 unique candidate–job pairs were assembled from the union of three recommendation methods' Top-5 outputs, supplemented with random jobs. Relevance was scored on an ordinal scale: 0 (irrelevant), 1 (partially relevant), and 2 (highly relevant). Two annotators, Aileen and Damir, scored the pairs independently; Jessy's ratings were subsequently reviewed from AI-assisted suggestions. We used median labels as provisional consensus rather than claiming an independently adjudicated gold standard.

We assessed annotation consistency with exact agreement and quadratic-weighted Cohen's kappa, and evaluated rankings using Precision@5 (only label 2 counted as highly relevant) and graded NDCG@5/10. Aileen and Damir agreed on 82.2% of the pairs, with weighted kappa of 0.816. This supports preliminary consistency of relevance judgments, but does not by itself demonstrate model accuracy.

The hybrid method (text weight 0.5) achieved higher NDCG@5 than TF-IDF for Finance (0.650 vs 0.516) and Healthcare (0.670 vs 0.463), but lower NDCG@5 for Data Science (0.596 vs 0.798). The skill-only method had incomplete rankings for two profiles because some jobs lacked detected skill evidence. The hybrid model is therefore not uniformly superior to the text baseline.

Qualitative error analysis exposed three failure patterns: highly relevant jobs demoted by hybrid scoring, irrelevant jobs promoted by high detected-skill coverage, and professionally regulated roles ranked highly without checking required licenses. These findings inform the user interface: show candidate skills and detected job skills separately, explain both score components, and distinguish unknown or unverified qualifications from confirmed ineligibility.

**Limitations.** The evaluation pool is small and partially selected using the models under study. The label set is provisional, and one rating source was AI-assisted and later reviewed by a human. Consequently, results support exploratory comparison and product diagnosis rather than broad claims of generalization, hiring suitability, or employment probability. All referenced postings are historical.

### Contribution to RQ2 and the Product

This work connects NLP feature extraction to an interpretable recommendation pipeline and tests whether higher algorithmic scores correspond to human-perceived relevance. The resulting diagnostics drive user-facing features: matched-skill explanations, missing or unverified skill indicators, configurable ranking methods, and explicit eligibility disclaimers. Together these features allow candidates to inspect why a role was recommended instead of treating a similarity score as a hiring probability.

## 4. Presentation Brief (2–3 minutes, English)

**Slide: What I built (30–40 seconds)**
"My contribution connects the job-description NLP pipeline to the recommendation evaluation. I standardized descriptions, extracted interpretable skill labels, and built a reproducible human relevance evaluation for three candidate profiles. This lets us compare ranking methods using human judgments rather than just model-generated similarity scores."

**Slide: What we found (50–70 seconds)**
"We evaluated 45 candidate–job pairs. The two independent human annotators agreed on about 82 percent of cases. Hybrid ranking improved NDCG@5 for the Finance and Healthcare profiles, but it performed worse than TF-IDF for Data Science. So adding skill overlap is not always beneficial. In one example, a highly relevant Senior Data Analyst role fell from rank 3 to rank 8 under Hybrid."

**Slide: Why it matters for users (40–50 seconds)**
"The main limitation is that detected skill overlap is not equivalent to real job suitability. A single detected matching skill can generate an inflated coverage score, and the current pipeline does not verify professional licenses. For the final website, we will show the matched skills, unverified skills, model score components, and historical-data limitations, so candidates can understand and challenge the recommendation."

**Close (10 seconds)**
"My evaluation turns the matching pipeline from a technical demo into a more transparent, testable product."

**Avoid saying:** "Our model accurately predicts job eligibility," "Hybrid is always better," or "Skill detection coverage measures extractor accuracy."

## 5. Website Product Definition

### Mandatory Page A — Historical Job Market Explorer

**User goal:** Understand the 2023–2024 job market before choosing career directions.

**Required controls:** job function, state/location (when present), job level/experience (when available), education (if grounded in extracted labels), salary range and salary-disclosure toggle (only when available). Each filter must visibly update results and denominators. Provide a **Reset Filters** action and show the number of jobs after filtering.

**Required content:** count of filtered historical postings; top skills by selected job function; salary distribution with disclosed-salary sample size; education/experience breakdown if supported; relevant charts/summaries from Aileen. For categorical filters, display explicit unknown/missing categories instead of implying absent requirements.

**Data:** cleaned job table (`data/processed/clean_jobs.parquet`) and Aileen's validated summary CSVs or on-demand cached aggregations. Do not rely solely on static PNGs for filters: those cannot update dynamically. Use sample data only as clearly labeled demo fallback.

**Acceptance:** altering the job function from Data Science to Finance updates displayed job count, skill demands, and applicable summaries; missing salary data is clearly disclosed.

### Mandatory Page B — Resume-to-Job Matching

**Primary user flow:**
1. Upload a PDF or DOCX resume (plus manual text fallback).
2. Extract machine-readable resume text; handle empty/scanned PDF with an explanatory error and text-paste fallback.
3. Run the existing `extract_skills()` on resume text, yielding **detected skills** only. Ask users to confirm/edit their skills, education, experience, and career interests before scoring. Do not claim full resume interpretation from keyword extraction.
4. Create the candidate profile dictionary expected by `score_jobs()` and `recommend_top_k()`.
5. Use the **2024 historical jobs subset** if the data contains a reliable date filter. If not, label the available sample as **historical 2023–2024 job postings** rather than claiming every result is from 2024.
6. Generate Top-K recommendations via the existing fitted matcher and ranking API. Provide method switch (TF-IDF / Hybrid baseline), Top-K input, and job-function/location filters. For fair ranking, score the fitted jobs first and then filter the scored table; do not pass a differently filtered table to `score_jobs()` with the original fitted matcher.
7. Show an expandable explanation per job: candidate-confirmed skills; extracted/detected job skills; matched skills; unverified job skills; text score; skill coverage score (or unknown); hybrid score; job title, role function, and available location/salary metadata; a short template-based recommendation explanation.
8. Show independent Career Paths / category suggestions using `rank_job_categories()` and the available `job_function` links.
9. Clearly label all recommendations as historical examples, not open roles, and all similarity scores as relevance signals, not hiring/eligibility probabilities.

**MVP resume parsing approach:** use `pypdf` to extract embedded PDF text, `python-docx` for DOCX paragraphs/tables. Add `pypdf`, `python-docx`, `streamlit` to requirements during the Thanksgiving implementation phase. Reject oversized input; do not save resumes to the repository or logs. No OCR, LLM, or external APIs are required for the MVP.

**Useful modules to add during Thanksgiving:**
- `src/resume_parser.py`: `extract_resume_text(uploaded_file)` and validation/error reporting; no network calls.
- `src/candidate_profile.py`: `detect_resume_skills(text)` and `build_candidate_profile(confirmed_inputs)`.
- `app.py`: Streamlit navigation, filters, display and caching; reuse existing `src/` ranking APIs unchanged.
- `tests/test_resume_parser.py`: normal PDF/DOCX, empty text, unsupported input and validation.
- `tests/test_app_workflow.py` or a documented smoke-test procedure: three candidate profiles and filter behavior.

**Page layout:** left sidebar with mode/filters; main page top with uploaded resume/profile confirmation; then recommended-job cards with evidence and expandable full JD; below, a career-category view. Scores should not be styled as probabilities.

**Acceptance:** test a selectable text PDF/DOCX resume and a text fallback; verify extracted skills are editable; run Top-5; explain a relevant and irrelevant job; filter by role and location; demonstrate that no output claims a 2024 posting is active.

### Optional Feature — Resume Improvement Assistant (Only After Both Pages Work)

**User goal:** Explain how to better present real experience for selected target jobs.

**MVP:** Display frequently requested detected skills among the selected target jobs that are absent from the *resume text*, not necessarily absent from the candidate's real experience. Ask the user which skills they actually possess; suggest truthful places to evidence those skills in the resume, without inventing qualifications. Provide specific, evidence-linked suggestions ("show a project demonstrating SQL"), not fabricated accomplishments. Do not train or call an LLM.

**Fallback:** If time is limited, display a simple skill-gap table and label resume rewrite as future work.

## 6. End-to-End Product Data Flow

`Resume PDF/DOCX -> Extract text -> Dictionary skill detection -> User edits/validates candidate profile -> Score historical jobs (TF-IDF / skill / Hybrid) -> Filter ranked output -> Top-K job cards + why-matched explanations -> Career category ranking -> Relevant market context -> Optional resume improvement suggestions`

**Data contracts:**
- Candidate input: `{candidate_id, skills: list[str], education: str, experience: str}` (user-edited).
- Job input: unique `job_id`, `title`, `description`, optional `job_function`, `location`, salary/experience/education fields.
- Scored-job output: `text_score`, `skill_score` (nullable), `hybrid_score`, `matched_skills`, `unverified_skills`, and `hybrid_uses_skills`.
- Career links: `{job_id, category}` with one link per job/category combination.

## 7. Coordination and Ownership

**Before next meeting:** Jessy submits three evaluation figures and the findings section; Damir submits weight sensitivity, Top-5 stability, and skill diagnostics; Aileen submits candidate-centered EDA figures with summary tables and data caveats. Keep core model source unchanged.

**Thanksgiving build:** Jessy implements two Streamlit pages and optional resume suggestions; Aileen owns presentation slides and exports candidate-centered data/figures; Damir owns written report and helps validate matching/category outputs.

**Final QA:** consistent job IDs and filters; same defined data window across pages; no hidden non-reproducible local dependencies; `python -m pytest tests -q`; full end-to-end demo; backup screenshots and prepared sample resume; clear historical-data and eligibility disclosures.

## 8. Priority If Time Is Tight

**P0:** Market Explorer filters with dynamic counts/skills; resume extraction + editable skills; real Top-K matching + reason explanations; reproducible final evaluation charts.  
**P1:** Career Paths, salary plots, improved layout and robust input errors.  
**P2:** Resume rewriting suggestions and alternative weighting controls.  

Do not sacrifice a working resume-to-Top-K path or functional market filters for extra features.
