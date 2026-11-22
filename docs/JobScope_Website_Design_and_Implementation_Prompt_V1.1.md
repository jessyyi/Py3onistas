# JobScope Website — Design Specification & Implementation Prompt

**Version:** 1.1 (provisional)  
**Date:** 2026-11-22  
**Design reference:** Refined Option E — Minimal Smart Filters  
**Project:** NYU DS-GA 1007, Team Py3onistas  
**Target branch:** `jessy-nlp`  
**Status:** Design approved provisionally; implementation and deployment not implied.

## Product goal

Build a functional, polished Streamlit application using the existing Python modules and historical LinkedIn U.S. job postings (2023–2024).

JobScope helps a user (1) explore historical job-market requirements, (2) review skills extracted from a resume, (3) discover relevant historical jobs, (4) understand the evidence behind the ranking, and (5) identify skills worth verifying or developing.

Historical jobs **must not** be described as active openings; recommendation scores are **relevance signals, not hiring or eligibility probabilities**.

## Team ownership and integration

- **Aileen — Market Intelligence:** cleaned data, EDA, validated market statistics, and plots. She selects and validates primary research charts and passes reusable data functions or aggregated tables to the website.
- **Damir — Recommendation:** TF-IDF, skill-only/hybrid scoring, Top-K, category ranking, weight-sensitivity study, model diagnostics. Do not silently modify his algorithms.
- **Jessy — NLP, Evaluation, Website:** JD/skill extraction, human evaluation, Streamlit implementation, UX, testing, screenshots, and demo. Integrate Aileen's and Damir's modules instead of duplicating their work.

## Design system — Refined Option E

- Premium minimal SaaS interface; spacious, consistent cards, neutral backgrounds, readable typography, restrained borders/shadows.
- Two top-level pages only: **Market Explorer** and **Resume Matching**.
- Consistent top navigation and a global theme selector: **Violet, Ocean, Sage, Sand**. Sage is the provisional default.
- Theme choice modifies accent colors, buttons, selections, chart palette, and subtle surfaces without changing data or ranking.
- Responsive desktop-first layout. Avoid decorative login/profile/notification features that do not actually exist.
- No process stepper, mandatory multi-page wizard, or large permanent filter sidebar.

## Page 1 — Historical Market Explorer

### Information architecture

1. Page heading, short description, visible historical-data label.
2. **Compact primary filter row**, ideally Job Function, Location, Industry, and one context-dependent filter. Applied filters appear as removable chips.
3. **Progressive Advanced Filters:** one compact button opens a nested accordion or popover (Compensation / Requirements / Company & Industry); sections are collapsed by default, and only supported fields appear. Include Reset All.
4. Filtered-sample count and derived KPI cards.
5. Interactive chart grid, informed by Aileen's actual validated EDA.

### Filtering and charts

Changing a filter must recompute KPI values and chart inputs from the filtered dataset. Display proper zero-result and unknown/missing-value states.

Candidate charts, pending Aileen's final work and available columns:
- top detected skills by function;
- salary distribution from valid, comparable salary fields;
- education and experience requirements;
- function and industry comparisons;
- location breakdown;
- other validated associations (e.g., skill/salary relationships), if interpretation is justified.

Never fabricate posting counts, trends, median salaries, skill premiums, or growth rates. Do not put illustrative mockup numbers into the actual application.

### Online lookup and external visuals — optional enhancement (P2)

The **primary charts come from the project's historical dataset**, not web searches. Chart rendering and hover details should use Plotly/Altair or equivalent interactive visualization over filtered data. This works without live internet access.

An optional supplementary panel can retrieve reliable **public** references through documented APIs or curated sources (e.g., occupational descriptions or separately dated labor-market benchmarks) if permitted and technically viable.

For every external item:
- Show the source name, URL, publication/reference period, retrieval timestamp, geography, and measurement definition.
- Clearly label it **External Context**, never merge it silently into the 2023–2024 LinkedIn sample.
- Handle missing network access and failed calls gracefully; cache responses and respect licensing, API limits, and privacy.
- Never use arbitrary web-search snippets as ground truth for the historical charts.
- Only use external images with appropriate reuse rights and attribution. Decorative imagery is not evidence.

This capability is deferred until core dashboard and resume matching are reliable.

## Page 2 — Job-centric Resume Matching

**One continuous vertical page.** Do not design a step-by-step wizard.

### Compact input and editable profile

- Upload text-based PDF or DOCX, or paste resume text.
- Extract text with pypdf/python-docx and existing `src/feature_extraction.py`.
- Show detected skill chips; allow manual additions/removals, education/experience confirmation, target function and preferred location.
- Handle blank documents, unsupported formats, large files, and scanned PDFs with helpful fallbacks.
- Do not persist, expose, or commit user resumes.

### Recommendations are the page's main focus

Once the candidate confirms the profile, render a **prominent vertical ranked list** of historical job cards. Offer Top-K = 5 / 10 / 20 and TF-IDF / Hybrid selectors. Use available occupation/location filters.

Each card contains, where available:
- Rank, job title, company, historical location, category, disclosed compensation;
- **Relevance Score** (not probability);
- component `text_score`, `skill_score`/Skill Coverage, and `hybrid_score`, with exact labels and a methodology tooltip;
- detected job skills, matched candidate skills, and unverified skills;
- a short **Why Recommended** explanation derived from actual text and skill evidence;
- expandable full historical job description.

Use displayed percentage only when defined as a clear transformation of the relevant score; never imply a 92% chance of being hired. Prefer raw normalized scores (e.g., 0.82 / 1) if the scale is clearer.

**Explainability rules:**
- Text similarity refers to TF-IDF-based similarity, not verified qualifications.
- Skill Coverage = matched detected job skills / all detected job skills.
- An absent detected skill is *unverified*, not proven missing.
- When zero job skills are detected, Skill Coverage is **Unknown / NaN**, not zero.
- Preserve the existing Hybrid fallback to TF-IDF when skill evidence is unavailable.
- Do not claim education, experience, licensing, eligibility, or employer demand has been verified unless the system actually verified it.

### Skill improvement and alternative career paths

Below (or adjacent to) the recommendations:
- **Skills to Build Next:** aggregate unverified requirements from relevant target jobs; show skill frequency, sample denominator, candidate status, and sensible evidence-based practice actions. No fabricated abilities or guarantee that acquiring a skill changes hiring odds.
- **Alternative Career Paths:** use `rank_job_categories` on the existing scored jobs, independently of the user's selected target category; explain category-level ranking separately from individual job-level relevance.
- Optional CSV export and synthetic sample profiles once core functionality works.

## Technical constraints

Existing modules to inspect and reuse:
- `src/data_processing.py`, `src/market_analysis.py`, `src/market_plots.py`;
- `src/feature_extraction.py`, `src/matching.py`, `src/recommendation.py`, `src/evaluation.py`.

**First audit the actual current files and signatures on `jessy-nlp`.** Plans do not prove a function exists.

`TextMatcher.fit(jobs)` and `score_jobs(candidate, jobs, matcher=...)` must receive the same underlying jobs DataFrame. For display filters, score the fitted table then filter the results, unless an existing reviewed interface supports an equivalent safe approach.

Cache shared job data and fitted matching resources where safe. Use a reproducible public/demo dataset for hosting and do not commit raw restricted data or personal resumes.

## Test and acceptance criteria

### Market Explorer
- Compact filters, removable chips, nested advanced filters, Reset All.
- Function/location filters produce real KPI and chart changes.
- Salary units and missingness correctly handled.
- Zero-results state, chart tooltips, clear dataset scope.

### Resume Matching
- PDF / DOCX / paste-text workflows succeed.
- Skills are editable and normalization consistent.
- TF-IDF and Hybrid Top-K work with real modules.
- Every score is correctly labeled and explained.
- Unknown skills do not become zeros.
- Three synthetic profiles complete the end-to-end flow.
- Category alternatives and evidence-based skill suggestions appear.

### Delivery
- Automated tests pass.
- Local Streamlit app starts successfully.
- Public deployment (if attempted) runs without local-only files.
- Provide demo screenshots, architecture diagram, feature summary, known limitations, and fallback demo for presentation.

## Work plan

**Phase 0 — Audit:** inspect branch, repository structure, current app, requirements, data schema, signatures, test status, and blockers.

**Phase 1 — App foundation:** navigation, theme system, common components, cached dataset loader, minimal skeleton.

**Phase 2 — Market Explorer:** compact filters, recomputed statistics, validated charts; collaborate with Aileen on final chart selection.

**Phase 3 — Resume Matching:** robust parser, editable candidate profile, real TF-IDF/Hybrid integration, job-centric explainable cards.

**Phase 4 — Career insights and polish:** skill-gap aggregation, career-category ranking, edge states, accessibility, responsiveness.

**Phase 5 — QA and deployment:** tests, synthetic demo profiles, performance/security checks, reproducible public demo if feasible.

**Phase 6 — Report and PPT handoff:** screenshots, system diagram, reproducible evidence, findings and limitations.

Provisional internal build window: 2026-11-22 to 2026-11-29. Final report: 2026-12-03; presentation: 2026-12-04 (verify with course announcements).

## Reusable implementation prompt

> You are my senior Python engineer, Streamlit developer, product designer and technical collaborator for JobScope (NYU DS-GA 1007, Team Py3onistas).
>
> Repository: https://github.com/jessyyi/Py3onistas; existing development branch: `jessy-nlp`. Local checkout: `/Users/jessy/Documents/nyu/career/github_project/Py3onistas`.
>
> Use this document as the authoritative design and scope for **Option E V1.1**. Communicate primarily in Chinese, but write all code, comments, identifiers, UI labels, documentation and Git commit messages in English.
>
> Inspect the **real current branch** before implementing. Reuse Aileen's market-analysis work and Damir's scoring/category algorithms rather than rewriting them.
>
> Build a two-page Streamlit app. Page 1: compact progressive nested filters, genuinely recalculated historical-market charts and transparent sample counts. Page 2: one vertically scrolling, **job-centric** resume-matching page with editable profile, Top-K recommendations, labeled relevance scores, actual evidence-based recommendation reasons, unverified skills, skill-gap suggestions, and independently ranked alternative careers. Do **not** create a process wizard.
>
> Implement global Violet/Ocean/Sage/Sand themes. Primary plots must come from verified historical data; any online references are optional, clearly sourced and separately labeled external context.
>
> Preserve NaN/Unknown skill behavior, TF-IDF fallback and the matcher DataFrame constraint. Never call relevance score an employment probability, or historical jobs active vacancies.
>
> Start with repository audit, then foundation, market explorer, resume matching, polish, tests and deployment. For each phase describe existing code, minimal changes, exact file paths and code/patches, verification commands, risks, and files to commit. Proceed with implementation instead of only planning. Do not merge into `main` until reviewed and tested.

---

**Design governance:** This is the current provisional website specification. Aileen determines which validated EDA charts are ready; Damir's final sensitivity findings determine whether to recommend a different model weight. Any deviation should be documented before implementing it.
