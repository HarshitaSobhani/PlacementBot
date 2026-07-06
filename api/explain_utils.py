"""Turns raw SHAP contributions into plain-language, per-prediction explanations.

Numeric/categorical column names are the caller's responsibility to supply
(from model/metadata.json, the single source of truth also used by
model/train.py) rather than hardcoded here, so the two can't drift apart.
"""

FRIENDLY_NAMES = {
    "Age": "age",
    "CGPA": "CGPA",
    "Internships": "number of internships",
    "Projects": "number of projects",
    "Coding_Skills": "coding skills rating",
    "Communication_Skills": "communication skills rating",
    "Aptitude_Test_Score": "aptitude test score",
    "Soft_Skills_Rating": "soft skills rating",
    "Certifications": "number of certifications",
    "Backlogs": "number of backlogs",
    "skill_avg": "overall skills average (coding/communication/soft skills)",
    "Gender": "gender",
    "Degree": "degree",
    "Branch": "branch",
}


def split_transformed_name(name: str, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[str, str | None]:
    """'num__CGPA' -> ('CGPA', None); 'cat__Branch_CSE' -> ('Branch', 'CSE')."""
    body = name.split("__", 1)[1]
    if body in numeric_cols:
        return body, None
    for col in categorical_cols:
        if body.startswith(col + "_"):
            return col, body[len(col) + 1 :]
    return body, None


def describe_feature(name: str, direction: str, raw_row: dict, numeric_cols: list[str], categorical_cols: list[str]) -> str:
    col, _category = split_transformed_name(name, numeric_cols, categorical_cols)
    friendly = FRIENDLY_NAMES.get(col, col)
    verb = "increased" if direction == "increased" else "decreased"
    value = raw_row.get(col)
    return f"Your {friendly} ({value}) {verb} your placement likelihood."
