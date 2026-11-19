# JobScope — Week 2 Update, Thursday Meeting Agenda & Final Delivery Plan

**Team:** Py3onistas — Jessy, Aileen, Damir  
**Course:** NYU DS-GA 1007 · Fall 2026  
**Updated:** November 17, 2026  
**Next team meeting:** Thursday, November 19, 2026  
**Status:** Working plan for team alignment; owners and meeting date to be confirmed on November 19

---

## 1. Message to the Team

Hi Aileen and Damir,

I wanted to share a consolidated update before our Thursday meeting and propose a clear plan for the remaining two weeks of JobScope. Our core data, NLP, and matching baselines are now working. I suggest that **Thursday, November 19 be our final major feature-planning and integration meeting**, so we can settle outstanding technical tasks, confirm interfaces, and assign the final deliverables. During Thanksgiving week, we can then work in parallel on the website/demo, presentation slides, and final written report.

The official in-person presentation sessions are **Friday, December 4, and Monday, December 7, between 4:00 and 9:00 PM ET**. We are **tentatively planning for December 4**, but we must confirm our group's assigned session/time and venue. We should finish the written report submission **before December 3**, and complete a full mock presentation before both the report submission and our assigned presentation session.

Here is the proposed progress update and task breakdown. Please review before Thursday and flag any blockers or changes needed.

## 2. Progress So Far

### Jessy — NLP Integration and Evaluation (completed/prepared)

- Merged the latest `main` locally and integrated NLP skill extraction with the cleaned 500-posting sample and the hybrid matching pipeline.
- Ran a complete candidate-to-Top-10 recommendation example.
- Processed **500 jobs**; **306 (61.2%)** received at least one detected skill. This is **skill-detection coverage, not precision or recall**.
- Added NLP-to-Matching integration tests and evaluation metric tests; **47 local tests passed** as of November 17.
- Generated an NLP error-analysis report and identified a potential hybrid-scoring failure mode: a job with only one detected matching skill can receive a perfect skill-coverage score despite weak text relevance.
- Created a shared **0/1/2 human relevance rubric** and a reproducible sample of **45 candidate–job pairs** across **three synthetic profiles** (random seed: 1007).
- Prepared separate blinded annotation files for Jessy and Aileen; the **actual human ratings have not yet been collected**.
- Implemented Precision@K, NDCG@K, and inter-rater agreement functions; **real ranking-quality results are still pending labels**.
- Opened **[PR #12 — Integrate NLP matching and prepare human evaluation](https://github.com/jessyyi/Py3onistas/pull/12)** for team review; it is **not yet merged**.

### Aileen — Data Cleaning and EDA (confirmed baseline / next delivery)

- Completed the data-cleaning pipeline, audit, reproducible 500-job sample, and initial EDA/salary-disclosure chart.
- Still to confirm: finalized market-analysis figures, occupation–skill heatmap, and 2–3 presentation-ready insights.
- Next evaluation role: independently label the same 45 candidate–job pairs as Jessy, without access to model scores or Jessy's ratings.

### Damir — Matching and Recommendation (confirmed baseline / next delivery)

- Completed the TF-IDF, skill-only, and hybrid Top-K recommendation baseline, tests, and a three-profile demonstration.
- Still to confirm: independent **career-category recommendations** (including category-size normalization and an alternative-career example), integration with Jessy's extraction output, and any adjustments needed for single-skill score inflation.
- Support the final ranking comparison and interface review.

> **Status note:** Aileen's and Damir's outstanding items above describe the planned deliverables that have not yet been confirmed in the shared GitHub repository. They may have unpushed local work.

## 3. Work to Finish by Thursday, November 19

**Jessy**
- [ ] Request review of PR #12; confirm changed files and merge status.
- [ ] Prepare a short live or notebook fallback demo: candidate profile → extracted skills → Top-K jobs.
- [ ] Present the 61.2% detection statistic, 47 passing tests, and one hybrid-ranking error example.
- [ ] Check that annotation files are easy for both raters to read and that the blinded evaluation protocol is clear.

**Aileen**
- [ ] Share the current EDA notebook and market figures, including sample sizes, salary units, and missingness.
- [ ] Select 2–3 defensible findings for the presentation.
- [ ] Confirm whether the occupation–skill heatmap is ready and when the remaining charts will be finalized.
- [ ] Agree to the independent annotation handoff and target date.

**Damir**
- [ ] Show progress on the career-category recommendation feature and identify any blockers.
- [ ] Verify compatibility with the integrated NLP skill labels and Top-K pipeline.
- [ ] Discuss the single-skill perfect-coverage issue and a lightweight score/weight sensitivity comparison.
- [ ] Agree on the final model comparison setup and a stable demo interface.

**All three**
- [ ] Decide which features will be included in the final product and which are out of scope.
- [ ] Set the feature-freeze goal, owner-specific handoff files, and GitHub PR review responsibilities.
- [ ] Confirm final report requirements and exact LMS submission deadline from the course announcement.
- [ ] **Choose the exact date/time for one final team meeting during December 1–3.**
- [ ] Confirm the team's assigned in-person presentation date, time slot, and location once released.

## 4. Thursday, November 19 — Proposed Meeting Agenda

| Time budget | Topic | Expected decision/output |
|---|---|---|
| 5 min | Jessy: NLP integration, tests, evaluation preparation | Review PR #12 and agree on next evaluation steps |
| 5 min | Aileen: EDA and figures | Final list of charts and 2–3 key market insights |
| 5 min | Damir: Top-K and career-category recommendations | Confirm demo-ready features and remaining implementation |
| 10 min | End-to-end integration review | Freeze interfaces, resolve blockers, assign remaining fixes |
| 10 min | Human evaluation + final deliverables | Annotation handoff, rubric, metric comparison, owner deadlines |
| 5 min | Presentation logistics | Pick Dec 1–3 meeting date; clarify submission/slot uncertainty |

**Meeting outcome:** an agreed feature list, updated issues/PR ownership, Thanksgiving deliverables, one scheduled final review, and a concrete mock-presentation plan.

## 5. Final Deliverables — Revised Ownership

| Owner | Primary responsibility | Expected deliverables | Support/Dependencies |
|---|---|---|---|
| **Jessy** | **Website / interactive demo** | Functional Streamlit app (`app.py`), candidate input, Top-K results, skill/score explanations, relevant market insights, stable demo script and fallback | Requires Aileen's cleaned data/figures and Damir's finalized recommendation APIs |
| **Aileen** | **Final PPT / slide design** | Complete slide deck, charts, diagrams, presenter notes/section assignments, polished design | Needs Jessy's screenshots/demo workflow, Damir's model explanation and report findings |
| **Damir** | **Final written report / document** | Final report draft, methods, model description, results, evaluation discussion, limitations, references, submission-ready document | Jessy supplies evaluation metrics/results; Aileen supplies EDA figures and market findings |

**Shared responsibility:** Everyone reviews factual claims, tests the app, validates slides/report alignment, rehearses, and presents their assigned section. Primary ownership does not remove the need for cross-review.

## 6. Thanksgiving Week — Parallel Work Plan (Nov 23–29)

**Jessy — Website / Demo**
- Implement the Streamlit app using the stable pipeline; do not prioritize unvalidated extra features.
- Show a synthetic candidate profile, Top-K job matches, detected/matched/unverified skills, and career-category suggestions if Damir's module is ready.
- Integrate 1–2 final EDA visualizations as appropriate, and show that postings are historical (2023–2024), not current job listings.
- Produce screenshots plus a deterministic, runnable backup (Notebook or recorded walkthrough if allowed).
- Independently complete the 45-pair human annotation; compute and share evaluation metrics once both raters finish.

**Aileen — Final PPT**
- Turn final market charts, recommendation outputs, and evaluation results into a concise presentation storyline.
- Cover problem/RQs, dataset, architecture, market analysis, matching/recommendation, evaluation, limitations, demo, and conclusion.
- Independently score the same 45 pairs by November 26 or the mutually agreed date; share the completed file privately for evaluation after both sets are finalized.
- Collect final visuals/screenshots from Jessy and methods/results from Damir.

**Damir — Final Report**
- Prepare the full written report, matching the original proposal's three research questions.
- Document dataset/cleaning, NLP, matching, career-category ranking, human evaluation, limitations, reproducibility, and contributions.
- Complete and test the career-category recommendation; review hybrid-ranking score sensitivity and send final outputs to Jessy and Aileen.
- Integrate Aileen's market charts and Jessy's evaluation results into the final report.

**Technical handoff target:** Core features and human evaluation should be substantially complete by **November 26** (feature freeze target). Presentation artifacts can be polished through November 29 and early December. If an owner has a blocker, notify the group before the handoff date.

## 7. December Wrap-Up, Submission and Presentation

| Date | Milestone | Owner(s) | Exit criterion |
|---|---|---|---|
| **Thu, Nov 19** | Final major feature-planning/integration meeting | All | Final scope, owners, blockers, handoffs, and Dec 1–3 meeting time decided |
| **Thu, Nov 26** | Target: human annotations and feature freeze | Jessy + Aileen (labels); Damir (model) | Labels complete and model/features stable enough for final comparison |
| **Nov 27–29** | Thanksgiving deliverables | Jessy website; Aileen slides; Damir final report | First complete versions shared with team |
| **Dec 1–3 (exact date TBD on Nov 19)** | **Final team wrap-up meeting** | All | Integrated app, PPT, report reviewed; remaining issues assigned |
| **Before report submission and Dec 4** | **Full mock presentation** | All | Timed end-to-end rehearsal, handoffs and backup demo checked |
| **Before Dec 3 (team target; official cutoff TBD)** | **Submit final report** | Damir leads, all review | Correct final file uploaded, submission confirmation captured |
| **Fri, Dec 4, 4–9 PM ET** | **In-person presentation session** | All | Tentative preferred session only — assigned time/location TBD |
| **Mon, Dec 7, 4–9 PM ET** | **Other in-person presentation session** | All as assigned | Attend at the assigned slot; optional viewing of other groups |

**Important schedule clarification:** The course notice identifies **December 4 and December 7** as in-person presentation dates, both between **4:00 and 9:00 PM ET**, with rooms and individual group slots to be assigned. **December 4 is our planning assumption, not a confirmed group assignment.** The notice quoted in chat does not establish an exact report-upload deadline; therefore our **internal target is to submit before December 3**, subject to the official LMS deadline.

**Sequencing requirement:** To allow a complete mock presentation **before report upload and December 4**, we should aim to hold the final wrap-up meeting and mock on **December 1 or 2**, rather than leave everything until December 3. The exact date will be selected on November 19.

## 8. Final Acceptance Checklist

- [ ] All relevant PRs reviewed and merged into `main` with tests passing.
- [ ] EDA visuals and 2–3 market insights finalized and accurately labeled.
- [ ] Top-K and career-category recommendation demos work from a reproducible environment.
- [ ] Both annotators complete independent judgments; agreement and ranking metrics documented, including caveats about the limited judged pool.
- [ ] Website is functional and has a fallback demonstration.
- [ ] Slides and report tell a consistent story and use the same numbers, figures, and terminology.
- [ ] Final report reviewed by everyone and uploaded before the internal deadline; submission verified.
- [ ] Full mock presentation completed, timed, and revised before the report upload and presentation.
- [ ] Official assigned presentation session, location, and time slot confirmed.

## 9. Links and Repository Conventions

- **Repository:** https://github.com/jessyyi/Py3onistas
- **NLP/evaluation PR:** https://github.com/jessyyi/Py3onistas/pull/12
- **Shared module interfaces:** `docs/SHARED_INTERFACES.md`
- **Evaluation rubric:** `docs/evaluation_rubric.md`
- **Existing project plan:** `docs/PROJECT_PLAN.md` (contains an older December 3 presentation assumption; refer to this document for the revised timetable)
- **This updated plan:** `docs/JobScope_Final_Delivery_Plan_2026-11-17.md`

**Git workflow:** Add this plan to the repository's `docs/` directory on the active feature branch, commit it separately, and merge it through a pull request. Avoid uploading raw job datasets, identifiable profiles, or unblinded annotation files to the public repository.
