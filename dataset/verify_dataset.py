import pandas as pd
import os

df = pd.read_csv("grogu_dataset/labels/master_labels.csv")
train = pd.read_csv("grogu_dataset/splits/train.csv")
val = pd.read_csv("grogu_dataset/splits/val.csv")
test = pd.read_csv("grogu_dataset/splits/test.csv")

print("=== DATASET VERIFICATION ===\n")

# 1. Class balance check
print("Class distribution (full dataset):")
print(df["class"].value_counts())
print()

# 2. Leakage check
train_ids = set(train["base_id"])
val_ids = set(val["base_id"])
test_ids = set(test["base_id"])
print(f"Train-Test base_id overlap: {len(train_ids & test_ids)} (must be 0)")
print(f"Val-Test base_id overlap: {len(val_ids & test_ids)} (must be 0)")
print(f"Train-Val base_id overlap: {len(train_ids & val_ids)} (must be 0)")
print()

# 3. File existence check
missing = 0
for _, row in df.iterrows():
    if not os.path.exists(row["pdf_path"]):
        missing += 1
print(f"Missing PDF files: {missing} (must be 0)")
print()

# 4. Temporal anachronism stats
print("Temporal anachronism delta distribution:")
print(df.groupby("class")["temporal_anachronism_delta"].describe())
print()

# 5. Metadata vector completeness
meta_df = pd.read_csv("grogu_dataset/metadata_vectors/all_metadata.csv")
print(f"Metadata vectors: {len(meta_df)} rows, {len(meta_df.columns)} columns")
print(f"Null values in metadata vectors: {meta_df.isnull().sum().sum()}")

print("\n=== VERIFICATION COMPLETE ===")
