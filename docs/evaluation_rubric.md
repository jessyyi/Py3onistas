# JobScope — Human Relevance Evaluation Rubric

**Project:** JobScope  
**Evaluation Lead:** Jessy  
**Second Annotator:** Aileen  
**Target Date:** November 26, 2026

## 1. Evaluation Objective

Evaluate how well JobScope recommends relevant job postings for a given candidate profile.

The evaluation compares three ranking approaches:

- TF-IDF text similarity
- Skill coverage matching
- Hybrid ranking

Human relevance judgments serve as an independent reference for ranking evaluation, not as evidence of hiring probability.

## 2. Relevance Scoring (0–2)

| Score | Label | Definition |
|---|---|---|
| 0 | Irrelevant | The job has little meaningful alignment with the candidate's skills, experience, or target career direction. |
| 1 | Partially Relevant | The job overlaps with some candidate skills or experience but has substantial mismatches in role responsibilities, specialization, or seniority. |
| 2 | Highly Relevant | The job closely aligns with the candidate's technical skills, background, and realistic career direction. |

### Scoring Considerations

Annotators should consider:

1. **Skill alignment:** Are the candidate's skills applicable to the role?
2. **Role relevance:** Do job responsibilities match the candidate's interests and professional background?
3. **Experience and education:** Is the expected seniority and education reasonably compatible?
4. **Transferability:** Could the candidate realistically transfer existing capabilities to the role?

Explicit mandatory qualifications may be noted separately. Missing information must not automatically be interpreted as ineligibility.

Annotators should assess overall relevance rather than assign arbitrary numerical subweights to the four dimensions.

## 3. Annotation Procedure

1. Select approximately 30–50 distinct candidate–job pairs across multiple candidate profiles and occupations, subject to feasibility.
2. Freeze the selected sample and record the sampling method and random seed.
3. Present identical candidate profiles and job descriptions to both annotators.
4. Jessy and Aileen assign relevance scores independently.
5. Hide all model similarity scores, ranking methods, and model-generated positions during annotation.
6. Record disagreements and calculate inter-rater agreement.
7. Reconcile disagreements only after independent labels have been saved.

## 4. Evaluation Metrics

### Precision@K

Convert human relevance labels into binary relevance:

- Score 2 = Relevant
- Scores 0 and 1 = Not highly relevant

Precision@K measures the fraction of the Top-K recommendations rated highly relevant.

Report K = 5 and K = 10 when the evaluation pool supports those cutoffs.

### NDCG@K

Use the original graded relevance labels (0, 1, 2).

NDCG@K measures how effectively the ranking places highly relevant jobs near the top.

Report NDCG@5 and NDCG@10.

Only calculate ranking metrics for candidate profiles with sufficiently complete relevance judgments over a defined candidate-job evaluation pool. Avoid treating unjudged jobs as irrelevant.

## 5. Model Comparison

Compare:

- TF-IDF-only ranking
- Skill-only ranking
- Hybrid ranking with text weight = 0.5

Investigate potential ranking problems:

- Jobs with only one detected skill receiving perfect coverage scores
- High similarity despite unrelated responsibilities
- Missing or incomplete extracted skill lists
- Jobs ranked highly despite inappropriate seniority

If time permits, conduct a sensitivity analysis using alternative hybrid weights.

## 6. Limitations

- Human relevance judgments are subjective.
- A small annotated sample cannot establish general performance across the entire U.S. job market.
- The skill vocabulary may omit occupation-specific competencies.
- Relevance does not establish eligibility, interview probability, or hiring outcomes.
- Historical job postings are used for research rather than current job availability.