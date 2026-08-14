import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)


# Remove unnecessary columns
# CustomerID is only an identifier.
# Unnamed: 0 is an unwanted index column.
df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split into training and testing sets
# Stratify keeps the 0/1 target proportion similar in both sets.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the split files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Xtrain shape: {Xtrain.shape}")
print(f"Xtest shape: {Xtest.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape: {ytest.shape}")
