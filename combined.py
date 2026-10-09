import pandas as pd

# ==========================================================
# LOAD DATASETS
# ==========================================================

healthy = pd.read_csv("healthy_features.csv")

apnea = pd.read_csv("apnea_features.csv")

# ==========================================================
# COMBINE
# ==========================================================

combined = pd.concat(
    [healthy, apnea],
    ignore_index=True
)

# ==========================================================
# SHUFFLE
# ==========================================================

combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================================
# CHECK DATASET
# ==========================================================

print("=" * 60)
print("COMBINED DATASET")
print("=" * 60)

print("\nHealthy Subjects :", len(healthy))
print("Apnea Subjects   :", len(apnea))
print("Total Subjects   :", len(combined))

print("\nColumns\n")
print(combined.columns.tolist())

print("\nMissing Values\n")
print(combined.isnull().sum())

print("\nClass Distribution\n")
print(combined["Label"].value_counts())

# ==========================================================
# SAVE
# ==========================================================

combined.to_csv(
    "combined_dataset.csv",
    index=False
)

print("\n")
print("=" * 60)
print("combined_dataset.csv SAVED SUCCESSFULLY")
print("=" * 60)

print("\nFirst Five Rows\n")
print(combined.head())