# Market EDA (RQ1)

Aileen's part. What U.S. job postings ask for, in the same order as a resume: education, experience, skills, then pay. The final charts go into the slides and the Market section of the website.

## How to run

1. Make the clean table once (Week 1):
   `python scripts/prepare_data.py --input <Kaggle folder> --output data/processed`
2. Run the tests: `python -m pytest tests -q`
3. Run `notebooks/01_data_eda.ipynb`. This is the full exploration. It also caches the text signals and the skill extraction in `data/processed/`, so the next notebook is fast.
4. Run `notebooks/04_market_story.ipynb`. This makes the 7 final charts in `figures/final/` and the summary tables in `data/summary/`.

`data/processed/` and `data/review/` stay on my computer (they have full descriptions). `data/summary/` only has counts and shares, so it is fine to push.

## Files

| File | What it does |
|---|---|
| `src/data_processing.py` | Week 1: loads the Kaggle tables and makes the clean table |
| `src/text_signals.py` | Reads years of experience, degree, majors, internship and research mentions from the description |
| `src/industry_groups.py` | Merges the 144 LinkedIn industries into plain groups (CS/IT, Finance, Health care, ...) |
| `src/market_analysis.py` | Counts, shares, salary summaries and skill statistics used in 01 |
| `src/resume_analysis.py` | Degree, majors, years and skills by group, used in 01 and 04 |
| `src/market_plots.py` | Charts for the exploration notebook (01) |
| `src/rq1_charts.py` | Tables and charts for the 7 final charts (04) |
| `tests/` | Tests on small made-up tables |

## Final charts (notebook 04)

| # | Chart | Summary table |
|---|---|---|
| 1 | Minimum degree by industry | `rq1_min_degree_by_industry.csv` |
| 2 | Top 3 majors by industry | `rq1_majors_by_industry.csv` |
| 3 | Years asked by LinkedIn level | `rq1_years_by_level.csv` |
| 4 | Median years by industry and level | `rq1_years_by_industry_level.csv` |
| 5 | Top skills by industry | `rq1_skills_by_industry.csv` |
| 6 | Job functions that share postings | `rq1_function_pairs.csv` |
| 7 | Median posted salary by industry and state | `rq1_salary_by_industry_state.csv` |

The PNGs are in `figures/final/` with the same numbers (`1_min_degree_by_industry.png`, ...). `rq1_takeaways.csv` has one sentence per chart (columns: chart, title, takeaway), and `industry_groups.csv` lists which LinkedIn industries are in each group.

### For the website

The app can read the files in `data/summary/` directly, without the full data:

- show `figures/final/<n>_*.png` with the sentence from `rq1_takeaways.csv` under it, in the order 1 to 7
- the Industry filter can use `industry_groups.csv` so the website uses the same groups as the charts

## Rules I used everywhere

- Industry = the hiring company's LinkedIn industry, merged into groups. A posting tagged twice in the same group counts once.
- Staffing and Recruiting is left out of the industry charts, because those firms post jobs for other companies.
- Job function and industry can have several labels. A posting counts once under each label, so shares across labels can add up to more than 100%.
- If a posting does not name a degree, a major or years, it is left out of that share. Not stated is never read as not required.
- Years = the smallest number the posting names ("3-5 years" counts as 3).
- Salary = full-time postings with a yearly USD salary between $10,000 and $1,000,000. A missing salary is never treated as zero. This is posted pay, not the legal minimum wage.
- Small groups are left out or marked "too few". The cutoff is written in each chart's note.

## Choices that changed the charts

- I left Retail out of chart 4, because its LinkedIn levels do not follow years (Mid-Senior level asks fewer years than Entry level). The notebook prints the Retail numbers.
- I left Management + Manufacturing out of chart 6. Almost every Manufacturing posting also has the Management tag, so that pair is about tagging.
- In chart 5 I hide skills named in under 3% of postings, and I drop the skill "Manufacturing" because it is the industry name, not a skill.
- In chart 7 I use the 6 states with the most salary postings. With 10 states, most smaller cells had under 50 postings.
