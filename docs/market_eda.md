# Market EDA (RQ1): how the code works

Owner: Aileen. This covers the Week 2 to 4 market analysis.

## How to run

1. Make the clean table once (Week 1):
   `python scripts/prepare_data.py --input <Kaggle folder> --output data/processed`
2. Run the tests: `python -m pytest tests -q`
3. Open `notebooks/01_data_eda.ipynb` and run all cells. Charts are saved in `figures/`.

The skill step and the text step save their results in `data/processed/`
(`job_skills_extracted.parquet`, `text_signals.parquet`), so they only run once.
These files and `data/review/` are not pushed to GitHub.

## Files

| File | What it does |
|---|---|
| `src/market_analysis.py` | All the numbers: counts, shares, salary summaries, skill statistics |
| `src/text_signals.py` | Reads years of experience and degree level from description text |
| `src/market_plots.py` | All the charts, in one shared style |
| `tests/test_market_analysis.py` | Tests on a small made-up table |
| `notebooks/01_data_eda.ipynb` | Runs everything in order and saves the charts |

## Notebook sections, functions and charts

| Section | Question | Main function | Chart |
|---|---|---|---|
| 0 | How many postings per month? | `postings_over_time` | `postings_over_time.png` |
| 1.1 | Which job functions are largest? | `count_by_group` | `postings_by_function.png` |
| 1.2 | Which functions appear on the same posting? | `function_cooccurrence` | `function_cooccurrence.png` |
| 1.3 | Which industries hire which jobs? | `crosstab_share` | `industry_by_function.png` |
| 1.4 | Which companies post the most? | `company_concentration` | `top_companies.png` |
| 1.5 | Where are the jobs? | `rate_by_group` | `map_postings_by_state.png` |
| 2.1 | Run Jessy's skill extractor on all postings | `extract_skills_for_jobs` | none |
| 2.2 | Which skills does each function ask for? (skill heatmap) | `skill_share_by_function`, `signature_skills` | `skills_by_function.png` |
| 2.3 to 2.4 | Typical skills and skill pairs | `signature_skills`, `skill_pairs` | tables |
| 2.5 | Hand check of the extractor | none | `data/review/skill_review_sample.csv` |
| 2.6 | Which skills go with higher pay? | `skill_salary_premium` | `skill_salary_premium.png` |
| 3 | Years and degree asked, by level | `experience_paradox`, `mix_by_group` | `experience_by_level.png`, `degree_by_function.png` |
| 4.1 | Which functions show a salary? | `rate_by_group` | `salary_disclosure_by_function.png` |
| 4.2 | Do pay laws change disclosure? | `add_pay_transparency_flag` | `pay_law_waffle.png`, `map_salary_disclosure.png` |
| 4.3 | Salary by function and level | `salary_ranges` | `salary_range_by_function.png`, `salary_range_by_level.png` |
| 4.4 | Salary and disclosure by industry | `salary_ranges`, `rate_by_group` | `salary_range_by_industry.png` |
| 4.5 | How wide are posted ranges? | `range_width` | table |
| 4.6 | Do remote jobs pay differently? | `remote_pay_gap` | `remote_vs_onsite_pay.png` |
| 5.1 | Remote share by function | `rate_by_group` | `remote_by_function.png` |
| 5.2 | Applications per 100 views | `competition_by_group` | `competition_by_function.png` |
| 5.3 | Missing data by function | `missing_matrix` | `missing_by_function.png` |

## Rules used everywhere

- Job function and industry can have several labels. A posting counts once under each label.
- A missing salary is never treated as zero.
- Salary charts use full-time postings with a yearly USD salary between $10,000 and $1,000,000.
- Every chart shows the number of postings behind each value (n).
- Groups with too few postings are left out. The cutoff is printed in each chart's note.
- Pay law start dates: CO 2021-01-01, CA and WA 2023-01-01, NY 2023-09-17, HI 2024-01-01, MD 2024-10-01 (clockspot.com, checked Nov 17, 2026).
