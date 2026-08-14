
import pandas as pd

# Path to the dataset
DATA_PATH = "tourism_project/data/tourism.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Expected columns from the project data dictionary
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]
# Check expected columns
missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

print("Dataset registered successfully.")

print("All expected columns are present.")
print(f"\nDataset shape: {df.shape}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts())

print("\nDataset summary:")
print(df.describe(include="all"))
