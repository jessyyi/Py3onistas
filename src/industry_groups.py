"""Merge LinkedIn company industries into plain groups.

Our data has 144 LinkedIn industry names and a lot of them are the same field
for someone looking for a job. For example "IT Services and IT Consulting" and
"Software Development" are both CS/IT jobs, and Banking / Insurance /
Financial Services are all finance. If I keep them apart, the charts split one
field into many small rows, so I map every name to one group here.

Staffing and Recruiting stays its own group. Those companies post jobs for
other companies, so I leave them out of the industry charts.
"""

import numpy as np
import pandas as pd

STAFFING = "Staffing and Recruiting"

GROUPS = {
    "CS/IT": [
        "IT Services and IT Consulting", "Software Development", "Technology, Information and Internet",
        "Computer and Network Security", "Information Services", "Computer Games", "Mobile Gaming Apps",
        "Telecommunications", "Wireless Services", "Computer Networking Products",
        "Computer Hardware Manufacturing", "Computers and Electronics Manufacturing",
        "Semiconductor Manufacturing",
    ],
    "Finance": [
        "Financial Services", "Banking", "Insurance", "Accounting", "Investment Banking",
        "Investment Management", "Venture Capital and Private Equity Principals",
    ],
    "Health care": [
        "Hospitals and Health Care", "Mental Health Care", "Medical Practices", "Alternative Medicine",
    ],
    "Pharma and biotech": [
        "Pharmaceutical Manufacturing", "Biotechnology Research", "Medical Equipment Manufacturing",
    ],
    "Manufacturing": [
        "Manufacturing", "Motor Vehicle Manufacturing", "Industrial Machinery Manufacturing",
        "Machinery Manufacturing", "Automation Machinery Manufacturing", "Food and Beverage Manufacturing",
        "Beverage Manufacturing", "Dairy Product Manufacturing", "Tobacco Manufacturing",
        "Chemical Manufacturing", "Plastics Manufacturing", "Packaging and Containers Manufacturing",
        "Paper and Forest Product Manufacturing", "Glass, Ceramics and Concrete Manufacturing",
        "Textile Manufacturing", "Furniture and Home Furnishings Manufacturing",
        "Personal Care Product Manufacturing", "Sporting Goods Manufacturing",
        "Appliances, Electrical, and Electronics Manufacturing", "Railroad Equipment Manufacturing",
        "Shipbuilding", "Printing Services",
    ],
    "Aerospace and defense": [
        "Defense and Space Manufacturing", "Aviation and Aerospace Component Manufacturing",
    ],
    "Retail": [
        "Retail", "Retail Apparel and Fashion", "Retail Groceries", "Retail Office Equipment",
        "Retail Luxury Goods and Jewelry",
    ],
    "Wholesale": ["Wholesale", "Wholesale Building Materials", "Wholesale Import and Export"],
    "Hospitality, food and leisure": [
        "Hospitality", "Food and Beverage Services", "Restaurants", "Travel Arrangements", "Events Services",
        "Gambling Facilities and Casinos", "Recreational Facilities", "Wellness and Fitness Services",
    ],
    "Media, marketing and arts": [
        "Advertising Services", "Public Relations and Communications Services", "Market Research",
        "Book and Periodical Publishing", "Newspaper Publishing", "Broadcast Media Production and Distribution",
        "Media Production", "Online Audio and Video Media", "Animation and Post-production",
        "Entertainment Providers", "Spectator Sports", "Performing Arts", "Musicians", "Artists and Writers",
        "Writing and Editing", "Photography", "Graphic Design", "Design Services",
        "Museums, Historical Sites, and Zoos",
    ],
    "Construction": ["Construction", "Civil Engineering", "Architecture and Planning"],
    "Real estate": ["Real Estate", "Leasing Non-residential Real Estate"],
    "Education": [
        "Higher Education", "Education Administration Programs", "Primary and Secondary Education",
        "E-Learning Providers", "Professional Training and Coaching", "Libraries",
    ],
    "Government and public safety": [
        "Government Administration", "Public Safety", "Armed Forces", "Law Enforcement",
        "Administration of Justice", "Legislative Offices", "Public Policy Offices", "Executive Offices",
        "International Affairs", "International Trade and Development",
    ],
    "Transportation and logistics": [
        "Truck Transportation", "Freight and Package Transportation", "Airlines and Aviation",
        "Transportation, Logistics, Supply Chain and Storage", "Maritime Transportation",
        "Warehousing and Storage",
    ],
    "Energy and utilities": [
        "Utilities", "Oil and Gas", "Mining", "Renewable Energy Semiconductor Manufacturing",
        "Environmental Services",
    ],
    "Non-profit and social services": [
        "Non-profit Organizations", "Individual and Family Services", "Civic and Social Organizations",
        "Religious Institutions", "Fundraising", "Philanthropic Fundraising Services",
        "Political Organizations", "Think Tanks",
    ],
    "Business, legal and HR services": [
        "Business Consulting and Services", "Outsourcing and Offshoring Consulting",
        "Strategic Management Services", "Law Practice", "Legal Services", "Alternative Dispute Resolution",
        "Human Resources Services", "Research Services", "Nanotechnology Research",
        "Translation and Localization", "Government Relations Services",
    ],
    "Facilities and other services": [
        "Facilities Services", "Security and Investigations", "Consumer Services", "Veterinary Services",
    ],
    "Farming and fishing": ["Farming", "Fisheries"],
    STAFFING: [STAFFING],
}

# name -> group, so each lookup is one dict access
LOOKUP = {name: group for group, names in GROUPS.items() for name in names}


def group_industries(labels):
    """Turn one posting's list of LinkedIn industries into a list of groups.

    A posting tagged both "IT Services and IT Consulting" and "Software
    Development" should count once under CS/IT, so I use a set. Names that are
    not in GROUPS keep their own name (so nothing gets lost silently).
    """
    if not isinstance(labels, (list, tuple, np.ndarray)):
        return []
    return sorted({LOOKUP.get(name, name) for name in labels})


def add_industry_group(jobs):
    """Copy of jobs where industry holds the groups. The LinkedIn names move to industry_raw."""
    out = jobs.copy()
    raw = out["industry_raw"] if "industry_raw" in out.columns else out["industry"]
    out["industry_raw"] = raw
    out["industry"] = raw.apply(group_industries)
    return out


def unmapped_industries(jobs):
    """LinkedIn names in the data that are not in GROUPS yet (should be empty)."""
    names = jobs["industry_raw"].explode().dropna()
    return sorted(set(names) - set(LOOKUP))


def group_table():
    """GROUPS as a two column table, so the website can show what each group means."""
    rows = [(group, name) for group, names in GROUPS.items() for name in names]
    return pd.DataFrame(rows, columns=["industry_group", "linkedin_industry"])
