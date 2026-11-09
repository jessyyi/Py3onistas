# Data audit: LinkedIn Job Postings (2023 to 2024)

Owner: Aileen. Status as of Nov 8, 2026.

Source: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

To rebuild the clean table, download the dataset to your own computer and run:

```bash
python scripts/prepare_data.py --input <path to the Kaggle folder> --output data/processed
python -m pytest tests -q
```

The full run takes about 11 seconds after the first load (5.6 s load, 4.6 s clean, 5.8 s save on the first run).

## Result

The clean table `data/processed/clean_jobs.parquet` has **119,055 U.S. postings**, one row per `job_id`. Teammates load it with `src.data_processing.load_clean_jobs()`.

## Step by step

| Step | Postings | Removed | Rule |
| --- | --- | --- | --- |
| Raw `postings.csv` | 123,849 | | Every `job_id` is unique. |
| U.S. filter | 123,773 | 76 | Dropped locations such as "NAMER", Canada, Brazil, Philippines. |
| Exact copies | 119,055 | 4,718 | Same company, title, description and location as an earlier posting. |

12,746 postings share their company, title and description with an earlier posting. Only 4,718 of them are in the same place. The rest are the same job posted in several cities. These are kept as separate openings. Column `same_text_postings` counts how many postings share the same text. 10,897 kept postings belong to such multi-city groups.

## Location

| Format | Example | Raw postings | State known? |
| --- | --- | --- | --- |
| City, ST | New York, NY | 105,088 | yes |
| Country only | United States | 8,125 | no |
| Metro area | San Francisco Bay Area | 6,003 | no |
| State name | California, United States | 4,557 | yes |
| Other (dropped) | NAMER, Ottawa, ON | 76 | |

All 202 distinct metro names were checked by hand. All are in the U.S.

## Joins

- Job functions come from `jobs/job_skills.csv` and `mappings/skills.csv`. There are 35 broad job-function tags. They are not individual technical skills.
- A posting has 1 to 3 job functions. Labels are collapsed into one list per posting before merging, so the row count does not change. A test checks this.
- `industry` is the company industry from `companies/company_industries.csv`, as defined in the proposal. `job_industry` is the posting-level industry from `jobs/job_industries.csv`. Both are lists.
- 7,990 rows in `job_skills.csv` refer to jobs that are not in `postings.csv`. They are ignored.

## Missing values in the clean table

| Field | Postings missing |
| --- | --- |
| Job function | 1,740 |
| Company industry | 1,861 |
| Description (empty) | 6 |
| Salary | 83,760 (salary stated for 35,295, or 29.6%) |

Missing salaries stay empty. They are never set to 0.

## Salary

- 35,295 postings state a salary. 35,281 of them are in USD and have a yearly value in `salary_annual`.
- Pay periods in the raw data: YEARLY 20,628, HOURLY 14,741, MONTHLY 518, WEEKLY 177, BIWEEKLY 9.
- `salary_annual` uses the dataset's `normalized_salary`. It was checked against our own conversion (hourly × 2080, weekly × 52, biweekly × 26, monthly × 12). All USD postings matched within 1%.
- 15 raw postings use other currencies (EUR, CAD, BBD, AUD, GBP). They get no yearly USD value.

## First finding

29.6% of U.S. postings state a salary. The share differs by job function: Accounting/Auditing 47% (n = 4,801) and Legal 45% (n = 2,140) are highest, and Health Care Provider is lowest at 23% (n = 16,092). Figure: `figures/salary_disclosure_by_function.png`.

## Limitations

- Historical postings from 2023 to 2024. They are not current openings.
- LinkedIn postings may not represent the whole U.S. job market.
- About 70% of postings give no salary, so salary results describe only the postings that do.
- 13,518 clean postings have no state (country only or metro area), so state-level results use fewer postings.
