"""Turns raw SHAP contributions into plain-language, per-prediction explanations."""

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

NUMERIC_COLS = [
    "Age", "CGPA", "Internships", "Projects", "Coding_Skills",
    "Communication_Skills", "Aptitude_Test_Score", "Soft_Skills_Rating",
    "Certifications", "Backlogs", "skill_avg",
]
CATEGORICAL_COLS = ["Gender", "Degree", "Branch"]


def split_transformed_name(name: str) -> tuple[str, str | None]:
    """'num__CGPA' -> ('CGPA', None); 'cat__Branch_CSE' -> ('Branch', 'CSE')."""
    body = name.split("__", 1)[1]
    if body in NUMERIC_COLS:
        return body, None
    for col in CATEGORICAL_COLS:
        if body.startswith(col + "_"):
            return col, body[len(col) + 1 :]
    return body, None


def describe_feature(name: str, direction: str, raw_row: dict) -> str:
    col, _category = split_transformed_name(name)
    friendly = FRIENDLY_NAMES.get(col, col)
    verb = "increased" if direction == "increased" else "decreased"
    value = raw_row.get(col)
    return f"Your {friendly} ({value}) {verb} your placement likelihood."
