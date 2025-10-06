import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("train.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
acc = clf.score(X_test, y_test)
print(f"✅ Model trained! Accuracy: {acc:.2f}")

# Save model
joblib.dump(clf, "punch_model.pkl")
print("💾 Saved model as punch_model.pkl")
