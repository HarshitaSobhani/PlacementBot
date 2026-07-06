"""
Exploratory data analysis for PlacementLens (real Kaggle "Student Placement
Dataset"). Produces 5 plots into eda/figures/. Run: python eda/eda.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "eda" / "figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "placement_data.csv")
NUMERIC = [
    "Age", "CGPA", "Internships", "Projects", "Coding_Skills",
    "Communication_Skills", "Aptitude_Test_Score", "Soft_Skills_Rating",
    "Certifications", "Backlogs",
]

# 1. Class balance
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Placement_Status", hue="Placement_Status", palette="viridis", legend=False)
plt.title("Class Balance: Placed vs Not Placed")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_balance.png", dpi=120)
plt.close()

# 2. CGPA / Aptitude Test Score distributions by placement status
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ["CGPA", "Aptitude_Test_Score"]):
    sns.kdeplot(data=df, x=col, hue="Placement_Status", fill=True, common_norm=False, alpha=0.4, ax=ax)
    ax.set_title(f"{col} distribution by placement status")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_percentage_distributions.png", dpi=120)
plt.close()

# 3. Feature correlations (numeric)
plt.figure(figsize=(8, 7))
corr = df[NUMERIC].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Numeric Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_feature_correlations.png", dpi=120)
plt.close()

# 4. Placement rate by Internships and Backlogs count
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ["Internships", "Backlogs"]):
    rate = df.groupby(col)["Placement_Status"].apply(lambda s: (s == "Placed").mean())
    rate.plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_ylabel("Placement rate")
    ax.set_title(f"Placement Rate by {col}")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_internships_backlogs_placement_rate.png", dpi=120)
plt.close()

# 5. Placement rate by Branch
plt.figure(figsize=(6, 4))
rate2 = df.groupby("Branch")["Placement_Status"].apply(lambda s: (s == "Placed").mean()).sort_values()
rate2.plot(kind="barh", color="#55a868")
plt.xlabel("Placement rate")
plt.title("Placement Rate by Branch")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_branch_placement_rate.png", dpi=120)
plt.close()

print(f"Saved 5 figures to {FIG_DIR}/")
