# JobScope — NLP Skill Detection Error Analysis

**Dataset:** 500-job cleaned sample
**Method:** Dictionary-based skill extraction
**Evaluation period:** November 2026

## 1. Overall Detection Coverage

- Total jobs: 500
- Jobs with detected skills: 306
- Jobs without detected skills: 194
- Skill detection coverage: 61.2%

Detection coverage is not recall or extraction accuracy.
Missing labels do not establish missing job requirements.

## 2. Coverage by Job Function

| Job Function | Jobs | Detected | Coverage | Avg. Skills |
|---|---:|---:|---:|---:|
| Health Care Provider | 72 | 47 | 65.3% | 1.03 |
| Management; Manufacturing | 60 | 26 | 43.3% | 0.85 |
| Business Development; Sales | 45 | 22 | 48.9% | 1.09 |
| Information Technology | 42 | 32 | 76.2% | 2.05 |
| Other | 41 | 18 | 43.9% | 0.98 |
| Engineering; Information Technology | 32 | 28 | 87.5% | 2.53 |
| Finance; Sales | 17 | 12 | 70.6% | 1.41 |
| Accounting/Auditing; Finance | 15 | 15 | 100.0% | 3.73 |
| Administrative | 10 | 3 | 30.0% | 0.70 |
| Sales | 9 | 6 | 66.7% | 1.78 |
| Unknown | 9 | 6 | 66.7% | 0.89 |
| Analyst; Information Technology; Research | 7 | 5 | 71.4% | 2.14 |
| Management | 7 | 3 | 42.9% | 0.71 |
| Engineering | 6 | 5 | 83.3% | 1.83 |
| Human Resources | 6 | 3 | 50.0% | 1.00 |
| Marketing; Sales | 6 | 2 | 33.3% | 0.83 |
| Quality Assurance | 6 | 5 | 83.3% | 1.33 |
| Art/Creative; Design; Information Technology | 5 | 2 | 40.0% | 1.60 |
| Education; Training | 5 | 0 | 0.0% | 0.00 |
| Accounting/Auditing | 4 | 3 | 75.0% | 1.50 |
| Information Technology; Project Management | 4 | 4 | 100.0% | 3.25 |
| Management; Sales | 4 | 1 | 25.0% | 0.75 |
| Marketing | 4 | 4 | 100.0% | 2.00 |
| Marketing; Product Management | 4 | 4 | 100.0% | 3.00 |
| Customer Service | 3 | 3 | 100.0% | 1.33 |
| Customer Service; General Business | 3 | 0 | 0.0% | 0.00 |
| Customer Service; Sales | 3 | 0 | 0.0% | 0.00 |
| Information Technology; Management; Other | 3 | 1 | 33.3% | 0.33 |
| Legal | 3 | 0 | 0.0% | 0.00 |
| Marketing; Public Relations; Writing/Editing | 3 | 3 | 100.0% | 1.33 |
| Analyst; Finance | 2 | 1 | 50.0% | 1.00 |
| Business Development; Consulting; Sales | 2 | 1 | 50.0% | 0.50 |
| Consulting; General Business | 2 | 2 | 100.0% | 1.00 |
| Customer Service; Management; Marketing | 2 | 1 | 50.0% | 0.50 |
| Education | 2 | 0 | 0.0% | 0.00 |
| Engineering; Quality Assurance | 2 | 2 | 100.0% | 2.00 |
| Manufacturing; Production | 2 | 1 | 50.0% | 0.50 |
| Purchasing; Supply Chain | 2 | 2 | 100.0% | 1.50 |
| Accounting/Auditing; Administrative; Finance | 1 | 0 | 0.0% | 0.00 |
| Accounting/Auditing; Analyst; Consulting | 1 | 1 | 100.0% | 2.00 |
| Accounting/Auditing; Finance; Strategy/Planning | 1 | 1 | 100.0% | 7.00 |
| Administrative; Customer Service; Human Resources | 1 | 1 | 100.0% | 3.00 |
| Administrative; Finance | 1 | 0 | 0.0% | 0.00 |
| Administrative; General Business; Other | 1 | 0 | 0.0% | 0.00 |
| Administrative; General Business; Sales | 1 | 1 | 100.0% | 1.00 |
| Administrative; Information Technology | 1 | 0 | 0.0% | 0.00 |
| Administrative; Other | 1 | 0 | 0.0% | 0.00 |
| Administrative; Product Management | 1 | 1 | 100.0% | 1.00 |
| Advertising; Business Development; Sales | 1 | 1 | 100.0% | 1.00 |
| Analyst; Engineering; Research | 1 | 1 | 100.0% | 5.00 |
| Analyst; General Business; Information Technology | 1 | 1 | 100.0% | 5.00 |
| Analyst; Project Management; Writing/Editing | 1 | 1 | 100.0% | 1.00 |
| Business Development; Marketing; Product Management | 1 | 0 | 0.0% | 0.00 |
| Business Development; Research; Science | 1 | 1 | 100.0% | 5.00 |
| Business Development; Sales; Supply Chain | 1 | 1 | 100.0% | 1.00 |
| Consulting; Customer Service; Engineering | 1 | 1 | 100.0% | 3.00 |
| Consulting; Health Care Provider | 1 | 0 | 0.0% | 0.00 |
| Customer Service; Marketing; Public Relations | 1 | 0 | 0.0% | 0.00 |
| Design; Education; Writing/Editing | 1 | 0 | 0.0% | 0.00 |
| Design; Engineering; Information Technology | 1 | 1 | 100.0% | 2.00 |
| Design; Engineering; Manufacturing | 1 | 1 | 100.0% | 1.00 |
| Design; Research | 1 | 1 | 100.0% | 6.00 |
| Distribution; Manufacturing; Product Management | 1 | 0 | 0.0% | 0.00 |
| Distribution; Manufacturing; Production | 1 | 0 | 0.0% | 0.00 |
| Distribution; Supply Chain | 1 | 1 | 100.0% | 1.00 |
| Engineering; Finance; Research | 1 | 1 | 100.0% | 1.00 |
| Engineering; Information Technology; Other | 1 | 1 | 100.0% | 6.00 |
| Engineering; Management | 1 | 1 | 100.0% | 2.00 |
| Engineering; Management; Other | 1 | 1 | 100.0% | 1.00 |
| Engineering; Manufacturing; Production | 1 | 1 | 100.0% | 6.00 |
| Engineering; Project Management | 1 | 1 | 100.0% | 4.00 |
| Finance | 1 | 1 | 100.0% | 7.00 |
| Finance; Marketing; Product Management | 1 | 1 | 100.0% | 1.00 |
| Finance; Sales; Strategy/Planning | 1 | 0 | 0.0% | 0.00 |
| Health Care Provider; Public Relations; Quality Assurance | 1 | 1 | 100.0% | 2.00 |
| Information Technology; Legal | 1 | 1 | 100.0% | 1.00 |
| Manufacturing; Production; Science | 1 | 1 | 100.0% | 1.00 |
| Manufacturing; Production; Supply Chain | 1 | 1 | 100.0% | 4.00 |
| Marketing; Public Relations; Sales | 1 | 0 | 0.0% | 0.00 |
| Production | 1 | 1 | 100.0% | 1.00 |
| Project Management | 1 | 1 | 100.0% | 2.00 |
| Quality Assurance; Research | 1 | 1 | 100.0% | 3.00 |
| Research | 1 | 1 | 100.0% | 2.00 |
| Supply Chain | 1 | 1 | 100.0% | 1.00 |

## 3. Most Frequently Detected Skills

| Skill | Job Count |
|---|---:|
| Excel | 74 |
| Manufacturing | 45 |
| Patient Care | 39 |
| Project Management | 38 |
| Agile | 31 |
| SQL | 20 |
| CRM | 18 |
| Data Analysis | 15 |
| Salesforce | 14 |
| SAP | 14 |
| AWS | 13 |
| Python | 13 |
| Change Management | 13 |
| Strategic Planning | 13 |
| Process Improvement | 12 |
| CPR | 12 |
| Quality Assurance | 12 |
| Budgeting | 11 |
| JavaScript | 11 |
| Forecasting | 10 |

## 4. Zero-Skill Detection Examples

| Job ID | Title | Description Excerpt |
|---|---|---|
| 3635117665 | Montessori teacher for grades K-3 | Seeking an experienced Montessori teacher/trainer for a small, established primary school, grades K-3. Applicant will model, instruct, super |
| 3844654703 | Account Manager | Position Overview:Our company is thrilled to present an exciting career opportunity for a candidate with expertise in the construction indus |
| 3854810848 | Associate | Job Description: I am/we are looking for a highly qualified and experienced civil litigation attorney with four to seven years’ experience t |
| 3884428994 | Pool Remodel Salesman | Pool Remodel Salesman Pay Range: Commission OnlyWe are seeking an experienced and dynamic sales and design person. If you have a proven trac |
| 3884431637 | Rental Linen & Uniform Sales | Join a Thriving Sales Team at American Linen and Uniform Supply! Are you eager to forge a bright career in sales? At American Linen and Unif |
| 3884872004 | Sales Manager | Position Summary: Our Sales Manager has managing partnerresponsibilities for a team of 5-25 Executive Consultants in marketing J.Galt's SaaS |
| 3884920026 | Front Desk-Part Time | Sage Mountain in Thousand Oaks, CA, an Agemark Senior Living Community is hiring for a Front Desk Part-Time position! For the SIX consecutiv |
| 3885100219 | Sales Manager | Position Summary: Our Sales Manager has managing partnerresponsibilities for a team of 5-25 Executive Consultants in marketing J.Galt's SaaS |
| 3885857586 | Sterile Processing Instructor - Per Diem | Company Description At Brookline College, we are dedicated to our students’ success and committed to providing them with our best service fr |
| 3885858690 | ER Travel RN - Job Id: JO03715746 | Health Carousel Nursing and Allied Health is seeking RN for an assignment in Clovis, CA starting 4/15/2024, Est. Weekly Pay: $2,129.00 |

## 5. Single-Skill Detection Examples

| Job ID | Title | Detected Skill |
|---|---|---|
| 3815847830 | Staff Software Engineer (AHT) | Agile |
| 3862373833 | Construction Estimator | Budgeting |
| 3884432411 | Fluorescent Penetrant Inspector | Manufacturing |
| 3884442160 | General Dentist | CPR |
| 3884446482 | Product Data Hub Developer | SQL |
| 3884811061 | Software Engineer | JavaScript |
| 3884819790 | CNC Work Group 2 | Manufacturing |
| 3884827403 | DC Department Manager 2nd Shift | Inventory Management |
| 3884832148 | Flight Nurse | Patient Care |
| 3884853614 | Compounder (1st Shift) - Princeton, NJ | Manufacturing |

## 6. Known Recommendation Failure Mode

Skill coverage can equal 1.0 when only one skill was detected and the candidate possesses it.

Example observed in the integration demo:

- Job: Veterans Affairs Specialist
- Text similarity: approximately 0.0085
- Skill coverage: 1.0
- Hybrid score: approximately 0.5042

This creates a risk of promoting weakly related jobs.
The example does not prove that the entire ranking system performs poorly.

## 7. Manual Review and Next Steps

- [ ] Review zero-skill examples for possible omissions.
- [ ] Review single-skill examples for false positives.
- [ ] Identify occupation-specific vocabulary gaps.
- [ ] Compare text-only and hybrid ranking relevance.
- [ ] Validate ranking quality using independent labels.

## 8. Limitations

The analysis reports automated detection statistics, not manually verified precision or recall.
Job-function values are grouped as stored in the sample.
The sample may not represent the full U.S. job market.
