# JobScope — Human Evaluation & Recommendation Findings

**Project:** JobScope — Job Market Intelligence & Personalized Career Recommendation System  
**Team:** Py3onistas  
**Evaluation Date:** November 19, 2026  
**Research Question:** RQ2 — Personalized Job Matching

## 1. Evaluation Objective

The objective of this evaluation is to assess whether JobScope can identify and rank relevant historical job postings for candidates with different professional backgrounds.

We compare three recommendation approaches:

- **TF-IDF:** Text similarity between candidate profiles and job descriptions.
- **Skill-only:** Coverage of detected job skills by the candidate's declared skills.
- **Hybrid:** A weighted combination of TF-IDF similarity and skill coverage, using an initial text weight of 0.5.

Human relevance judgments are used to evaluate recommendation quality rather than treating model-generated matching scores as ground truth.

## 2. Evaluation Design

The evaluation contains **45 unique candidate–job pairs**, distributed evenly across three synthetic candidate profiles:

| Candidate Profile | Evaluated Job Pairs |
|---|---:|
| Data Science | 15 |
| Finance | 15 |
| Healthcare | 15 |
| **Total** | **45** |

The evaluation pool was constructed using the union of Top-5 recommendations from TF-IDF, Skill-only, and Hybrid ranking, supplemented with randomly selected jobs where necessary.

Each pair was scored on a three-level relevance scale:

- **0 — Irrelevant:** Little meaningful alignment with the candidate.
- **1 — Partially Relevant:** Some relevant skills or experience, but substantial mismatches.
- **2 — Highly Relevant:** Strong alignment with the candidate's background and career direction.

### Annotation Provenance

Aileen and Damir submitted independent annotations. Jessy's ratings originated from AI-assisted suggestions that were subsequently reviewed and accepted by Jessy.

The three rating sources were combined using the median relevance score for each candidate–job pair.

This produces a provisional consensus reference for exploratory evaluation. It should not be described as a fully independent three-human blind annotation study.

## 3. Annotation Reliability

| Comparison | Exact Agreement | Cohen's Kappa | Quadratic Weighted Kappa |
|---|---:|---:|---:|
| Jessy vs Aileen | 73.3% | 0.569 | 0.750 |
| Jessy vs Damir | 82.2% | 0.707 | 0.800 |
| Aileen vs Damir | 82.2% | 0.717 | 0.816 |

The three rating sources agreed completely on 31 out of 45 pairs (68.9%).

Aileen and Damir achieved 82.2% exact agreement and a quadratic-weighted Cohen's kappa of 0.816, providing useful preliminary evidence of annotation consistency.

Fourteen candidate–job pairs contained disagreements. All disagreements involved adjacent relevance levels, with no extreme 0-versus-2 conflicts.

The largest concentration of disagreement occurred in Healthcare, particularly for roles requiring professional licenses or specialized qualifications.

## 4. Recommendation Performance

### 4.1 NDCG@5

| Candidate | TF-IDF | Skill-only | Hybrid |
|---|---:|---:|---:|
| Data Science | **0.798** | N/A | 0.596 |
| Finance | 0.516 | 0.230 | **0.650** |
| Healthcare | 0.463 | N/A | **0.670** |

NDCG@5 measures whether highly relevant jobs appear near the top of the recommendation list.

The Hybrid approach outperformed TF-IDF for Finance and Healthcare, but underperformed TF-IDF for Data Science.

Skill-only results were unavailable for Data Science and Healthcare because missing detected-skill evidence resulted in incomplete ranked evaluation pools.

### 4.2 Precision@5

| Candidate | TF-IDF | Skill-only | Hybrid |
|---|---:|---:|---:|
| Data Science | 0.20 | N/A | 0.00 |
| Finance | 0.20 | 0.20 | **0.40** |
| Healthcare | 0.20 | N/A | 0.20 |

Precision@5 defines highly relevant jobs as those with a human relevance label of 2.

The Finance Hybrid model placed two highly relevant jobs within its top five recommendations.

For Data Science, the Hybrid model did not place any highly relevant jobs within the top five, even though partially relevant jobs were present.

These results indicate that Hybrid Matching does not consistently outperform the text-only baseline.

## 5. Ranking Error Analysis

Three representative cases illustrate the limitations of the current recommendation pipeline.

| Candidate | Job | Observation | Interpretation |
|---|---|---|---|
| Data Science | Senior Data Analyst | Highly relevant job moved from TF-IDF rank 3 to Hybrid rank 8 | Skill coverage may reduce the ranking of otherwise relevant jobs |
| Finance | Database Administrator | Irrelevant job appeared at Hybrid rank 5 | Detected skill overlap may inflate matching scores |
| Healthcare | Nurse Practitioner | Irrelevant job appeared at Hybrid rank 2 | Skill relevance does not establish professional eligibility |

### Skill Score Inflation

The existing skill coverage metric measures:

Skill Coverage = Matched Detected Job Skills / Total Detected Job Skills

A job with only one detected skill can receive a perfect score if that skill matches the candidate.

This can exaggerate matching quality when the skill dictionary fails to capture important requirements.

### Eligibility Limitations

The matching system does not independently verify professional licenses, required degrees, experience thresholds, or other mandatory qualifications.

Therefore, high matching scores should not be interpreted as evidence that a candidate is eligible for a position.

## 6. Product Implications

The evaluation findings inform several important JobScope features.

**Explainable Job Matching:** Recommendations should display text similarity, detected skill coverage, matched skills, and job skills that are not verified in the candidate profile.

**Skill Gap Identification:** The system should identify relevant job skills not confirmed in a candidate's resume. These should be presented as potential gaps for review, not proof that the candidate lacks the skills.

**Candidate Profile Confirmation:** Automatically extracted resume skills should be editable before matching.

**Eligibility Transparency:** The website should distinguish relevance from verified job eligibility, particularly for regulated professions.

**Historical Market Context:** Recommendations should be explicitly labeled as historical 2023–2024 job postings rather than currently available vacancies.

These features turn matching scores into information that users can interpret and act upon.

## 7. Limitations

This evaluation has several limitations:

1. The dataset contains only 45 judged candidate–job pairs from three synthetic profiles.
2. The evaluation pool is partly selected from the models being compared.
3. Jessy's labels were AI-assisted and subsequently reviewed.
4. The skill dictionary may omit important occupation-specific requirements.
5. Skill-only rankings can be incomplete when skill evidence is missing.
6. Mandatory eligibility conditions are not independently verified.
7. Hybrid weights have not been validated on a separate representative evaluation dataset.

Accordingly, the findings should be interpreted as an exploratory comparison rather than evidence of broad real-world recommendation performance.

## 8. Conclusion

The evaluation demonstrates that combining textual similarity and detected skill coverage can improve recommendation rankings for some candidate profiles, but does not guarantee better performance across occupations.

Hybrid Matching achieved higher NDCG@5 for Finance and Healthcare, while TF-IDF performed better for Data Science.

The error analysis also revealed the importance of skill extraction coverage, model-score transparency, and professional eligibility constraints.

These findings support JobScope's development as an explainable recommendation system that helps candidates explore historical job opportunities, understand relevant skills, and investigate alternative career paths.

Further research could evaluate alternative Hybrid weights and improved skill coverage using independent evaluation data.