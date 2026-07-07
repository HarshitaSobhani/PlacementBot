from api.explain_utils import describe_feature, split_transformed_name

NUMERIC = ["CGPA", "Backlogs"]
CATEGORICAL = ["Gender", "Branch"]


def test_split_transformed_name_numeric():
    assert split_transformed_name("num__CGPA", NUMERIC, CATEGORICAL) == ("CGPA", None)


def test_split_transformed_name_categorical():
    assert split_transformed_name("cat__Branch_CSE", NUMERIC, CATEGORICAL) == (
        "Branch",
        "CSE",
    )


def test_describe_feature_reads_raw_value_not_encoded_one():
    raw_row = {"CGPA": 8.1, "Branch": "CSE"}
    text = describe_feature("num__CGPA", "increased", raw_row, NUMERIC, CATEGORICAL)
    assert "8.1" in text and "increased" in text
