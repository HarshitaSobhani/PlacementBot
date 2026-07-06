from typing import Literal

from pydantic import BaseModel, Field


class PlacementInput(BaseModel):
    # Bounds are the real training data's min/max (see data/placement_data.csv)
    # plus a little headroom -- not arbitrary caps. The model has never seen
    # inputs outside this range, so predictions beyond it would be silent
    # extrapolation rather than something backed by training data.
    Age: int = Field(..., ge=17, le=26)
    Gender: Literal["Male", "Female"]
    Degree: Literal["B.Tech", "BCA", "MCA", "B.Sc"]
    Branch: Literal["ECE", "ME", "Civil", "CSE", "IT"]
    CGPA: float = Field(..., ge=4.0, le=10.0)
    Internships: int = Field(..., ge=0, le=5)
    Projects: int = Field(..., ge=0, le=8)
    Coding_Skills: int = Field(..., ge=1, le=10)
    Communication_Skills: int = Field(..., ge=1, le=10)
    Aptitude_Test_Score: float = Field(..., ge=30, le=100)
    Soft_Skills_Rating: int = Field(..., ge=1, le=10)
    Certifications: int = Field(..., ge=0, le=5)
    Backlogs: int = Field(..., ge=0, le=5)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 22,
                "Gender": "Male",
                "Degree": "B.Tech",
                "Branch": "CSE",
                "CGPA": 8.1,
                "Internships": 2,
                "Projects": 4,
                "Coding_Skills": 8,
                "Communication_Skills": 7,
                "Aptitude_Test_Score": 82,
                "Soft_Skills_Rating": 7,
                "Certifications": 2,
                "Backlogs": 0,
            }
        }
    }


class TopFeature(BaseModel):
    feature: str
    direction: Literal["increased", "decreased"]
    approx_probability_impact_pct: float
    explanation: str


class PredictionResponse(BaseModel):
    prediction: Literal["Placed", "Not Placed"]
    probability_placed: float
    top_features: list[TopFeature]
