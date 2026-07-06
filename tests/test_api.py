from fastapi.testclient import TestClient

from api.main import app
from api.schema import PlacementInput

client = TestClient(app)
EXAMPLE_PAYLOAD = PlacementInput.model_config["json_schema_extra"]["example"]


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_returns_expected_shape():
    res = client.post("/predict", json=EXAMPLE_PAYLOAD)
    assert res.status_code == 200

    body = res.json()
    assert body["prediction"] in ("Placed", "Not Placed")
    assert 0.0 <= body["probability_placed"] <= 1.0
    assert len(body["top_features"]) == 3
    for feature in body["top_features"]:
        assert feature["direction"] in ("increased", "decreased")
        assert isinstance(feature["explanation"], str) and feature["explanation"]
        # The public API contract should expose clean column names, not
        # internal ColumnTransformer prefixes like "cat__"/"num__".
        assert "__" not in feature["feature"]


def test_predict_rejects_out_of_range_input():
    payload = dict(EXAMPLE_PAYLOAD, CGPA=999)
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
