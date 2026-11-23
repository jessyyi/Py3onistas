"""Regression checks for web helper behavior using synthetic labels."""

import pandas as pd

from app import counts, subset, to_labels


def test_to_labels_parses_csv_list_and_plain_text():
    assert to_labels("['Data Science', 'Finance']") == ["Data Science", "Finance"]
    assert to_labels("Python") == ["Python"]
    assert to_labels(None) == []


def test_multilabel_filters_and_counts():
    jobs = pd.DataFrame({
        "job_id": [1, 2],
        "job_function": [["Data Science", "Finance"], ["Finance"]],
        "skill_labels": [["Python", "SQL"], ["SQL"]],
    })
    assert len(subset(jobs, "job_function", "Data Science")) == 1
    assert counts(jobs, "skill_labels").loc["SQL"] == 2
