import pandas as pd
import json
from pathlib import Path

# =========================
# Paths
# =========================
DATA_IN = Path("../data/obesity_ prediction.csv")
DATA_OUT = Path("../data/obesity_numeric.csv")
MAPPINGS_OUT = Path("../data/mappings.json")

# =========================
# Read the original data
# =========================
df = pd.read_csv(DATA_IN)
print("Original shape:", df.shape)

# =========================
# Create a copy for numeric encoding
# =========================
df_num = df.copy()
mappings = {}

# =========================
# Binary columns: yes/no -> 1/0
# =========================
binary_cols = ["family_history", "FAVC", "SMOKE", "SCC"]
for col in binary_cols:
    mappings[col] = {"no": 0, "yes": 1}
    df_num[col] = df_num[col].map(mappings[col])

# =========================
# Gender: Female -> 0, Male -> 1
# =========================
mappings["Gender"] = {"Female": 0, "Male": 1}
df_num["Gender"] = df_num["Gender"].map(mappings["Gender"])

# =========================
# Frequency columns
# =========================
freq_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
for col in ["CAEC", "CALC"]:
    mappings[col] = freq_map
    df_num[col] = df_num[col].map(freq_map)

# =========================
# MTRANS: categorical transport
# =========================
mtrans_map = {
    "Public_Transportation": 0,
    "Walking": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4
}
mappings["MTRANS"] = mtrans_map
df_num["MTRANS"] = df_num["MTRANS"].map(mtrans_map)

# =========================
# Obesity: target label
# =========================
target_map = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
}
mappings["Obesity"] = target_map
df_num["Obesity"] = df_num["Obesity"].map(target_map)

# =========================
# Add BMI column (optional)
# =========================
df_num["BMI"] = df_num["Weight"] / (df_num["Height"] ** 2)

# =========================
# Save the numeric CSV (semicolon separated)
# =========================
df_num.to_csv(DATA_OUT, index=False, sep=";")
print("Saved numeric data to:", DATA_OUT)

# =========================
# Save mappings
# =========================
with open(MAPPINGS_OUT, "w") as f:
    json.dump(mappings, f, indent=2)
print("Saved mappings to:", MAPPINGS_OUT)

# =========================
# Preview
# =========================
print(df_num.head(5))