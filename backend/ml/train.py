import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("dataset/train.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

# Accuracy
acc = clf.score(X_test, y_test)
print(f"✅ Model trained! Accuracy: {acc:.2f}")

# Save
joblib.dump(clf, "models/punch_classifier.pkl")
