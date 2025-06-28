import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox
import numpy as np

# Load dataset
# Use the correct relative path to the Data folder
# If running this script, make sure the working directory is the project root

data = pd.read_csv("./Data/heart_2020_cleaned.csv")

# Encode all object (string) columns automatically
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Drop rows with missing values if any
data = data.dropna()

# Features & Target
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']  # Already encoded as 0/1

# Ensure all features are numeric (for safety)
if not all([pd.api.types.is_numeric_dtype(X[col]) for col in X.columns]):
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col])[0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- GUI Section ---
feature_hints = {}
for col in X.columns:
    unique_vals = data[col].unique()
    if len(unique_vals) <= 10:
        feature_hints[col] = f"Possible values: {sorted([int(v) for v in unique_vals])}"
    else:
        feature_hints[col] = "Enter a numeric value"

def predict():
    try:
        vals = []
        for i, entry in enumerate(entries):
            val = entry.get()
            if val == '':
                raise ValueError(f"Empty input for '{labels[i]}'")
            vals.append(float(val))
        vals = np.array(vals).reshape(1, -1)
        pred = model.predict(vals)[0]
        result = "⚠️ High Risk of Heart Disease" if pred == 1 else "✅ Low Risk"
        messagebox.showinfo("Prediction", result)
    except Exception as ex:
        messagebox.showerror("Error", f"Please enter valid numbers for all fields.\n{ex}")

labels = list(X.columns)
entries = []
root = tk.Tk()
root.title("Heart Disease Predictor")
for i, label in enumerate(labels):
    hint = feature_hints[label]
    tk.Label(root, text=f"{label} ({hint})").grid(row=i, column=0, sticky='w')
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)
tk.Button(root, text="Predict", command=predict).grid(row=len(labels), column=0, columnspan=2)
root.mainloop()
