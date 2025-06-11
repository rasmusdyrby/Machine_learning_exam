import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("student_habits_performance.csv")
df = df.drop(columns=["student_id"])  # Drop irrelevant ID column

# Separate features and target
X = df.drop(columns=["exam_score"])
y = df["exam_score"]

# Identify categorical features for one-hot encoding
categorical_cols = X.select_dtypes(include="object").columns.tolist()

# Define preprocessing: OneHotEncoding for categoricals
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

algorithms = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

# Split data for training/testing ONCE so it's the same for both models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}  # store results for summary table

for name, regressor in algorithms.items():
    print(f"\n--- Training {name} ---")
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    pipeline.fit(X_train, y_train)

    # Save the trained pipeline model to disk
    filename = f"student_score_{name.lower()}.pkl"
    dump(pipeline, filename)
    print(f"Saved model to {filename}.")

    # Compute and save imputation values only ONCE
    # (You only need to save this once, but it is run for each model for simplicity)
    impute_vals = {}
    for col in X.columns:
        if X[col].dtype in ("float64", "int64"):
            impute_vals[col] = X[col].mean()
        else:
            impute_vals[col] = X[col].mode()[0]
    dump(impute_vals, "impute_defaults.pkl")

    # -- Evaluate performance using RMSE
    y_train_pred = pipeline.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"{name} Train RMSE: {rmse_train:.2f}")

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {
        "train_rmse": rmse_train,
        "test_rmse": rmse
        }  # save for summary
    print(f"{name} RMSE: {rmse:.2f}")

    # -- Plot: Actual vs. Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Exam Score")
    plt.ylabel("Predicted Exam Score")
    plt.title(f"{name}: Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    # -- Feature Importance
    if name == "RandomForest":
        importances = pipeline.named_steps["regressor"].feature_importances_
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        feat_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print("\nRandomForest Feature Importances (Top 10):\n", feat_df.head(10))
        plt.figure(figsize=(7,5))
        feat_df.head(10).plot(kind="barh", title="RandomForest: Top 10 Important Features")
        plt.xlabel("Importance Score")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    elif name == "LinearRegression":
        # Optional: Also plot LR "importance"
        try:
            # Find feature names after preprocessing
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            coefs = pipeline.named_steps["regressor"].coef_
            coef_df = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)
            print("\nLinear Regression Coefficient Magnitude (Top 10):\n", coef_df.head(10))
            plt.figure(figsize=(7,5))
            coef_df.head(10).plot(kind="barh", title="LinearRegression: Top 10 Absolute Coefficient Features")
            plt.xlabel("Coefficient Value")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("(Could not plot linear regression coefficients: ", e, ")")

# Print summary RMSE comparison
print("\nModel comparison (Train vs. Test RMSE):")
for model, res in results.items():
    print(f"{model:20}: Train RMSE = {res['train_rmse']:.2f}, Test RMSE = {res['test_rmse']:.2f}")