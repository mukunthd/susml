# =====================================================================
# SageMaker Unified Studio | End-to-End Workflow with MLflow Tracking
# =====================================================================

# 1. Imports & Environment ----------------------------------------------------
import pandas as pd
import numpy as np
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib, os, warnings
warnings.filterwarnings("ignore")

print("✔️  Libraries imported")

# 2. Configure Managed-MLflow --------------------------------------------------
TRACKING_SERVER_ARN = "your-tracking-server-arn"          # <── UPDATE ME
mlflow.set_tracking_uri(f"aws://sagemaker/{TRACKING_SERVER_ARN}")
mlflow.set_experiment("sales-prediction-experiment")
print("✔️  MLflow tracking configured")

# 3. Load Sample Dataset ------------------------------------------------------
# -- Upload sample_sales_data.csv to Studio first or load from Catalog
df = pd.read_csv("sample_sales_data.csv")
print(f"✔️  Dataset loaded  |  shape = {df.shape}")

# 4. Data Cleaning & Preparation ---------------------------------------------
# a. Remove duplicates
df = df.drop_duplicates()

# b. Impute missing values / blanks
df["product_name"] = df["product_name"].replace("", np.nan).fillna("Unknown")
mode_region = df["region"].replace("", np.nan).mode()[0]
df["region"] = df["region"].replace("", np.nan).fillna(mode_region)

# c. Remove rows with invalid quantity (≤0)
df = df[df["quantity"] > 0]

# d. Type conversions & derived features
df["order_date"] = pd.to_datetime(df["order_date"])
df["total_sales"] = df["price"] * df["quantity"]
df["month"]       = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.dayofweek

print("✔️  Data cleaned  |  shape =", df.shape)

# 5. Feature Engineering ------------------------------------------------------
median_sales = df["total_sales"].median()
df["high_value_customer"] = (df["total_sales"] > median_sales).astype(int)

le_product = LabelEncoder()
le_region  = LabelEncoder()
df["product_enc"] = le_product.fit_transform(df["product_name"])
df["region_enc"]  = le_region.fit_transform(df["region"])

features = ["price","quantity","month","day_of_week","product_enc","region_enc"]
X, y = df[features], df["high_value_customer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 6. Train & Track Models -----------------------------------------------------
def train_model(model, model_name, params:dict):
    with mlflow.start_run(run_name=model_name):
        # log parameters
        for k,v in params.items(): mlflow.log_param(k,v)
        mlflow.log_param("model_type", model_name)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        # metrics
        acc = accuracy_score(y_test, y_pred)
        prec= precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        mlflow.log_metrics({"accuracy":acc,"precision":prec,"recall":rec,"f1":f1})
        # log model
        mlflow.sklearn.log_model(model, model_name,
                                 registered_model_name=f"sales_{model_name}")
        print(f"{model_name}: acc={acc:.3f}  f1={f1:.3f}")
        return model, f1

rf_params = {"n_estimators":100,"max_depth":10,"random_state":42}
rf_model, f1_rf = train_model(RandomForestClassifier(**rf_params),"random_forest",rf_params)

lr_params = {"C":1.0,"solver":"liblinear","random_state":42}
lr_model, f1_lr = train_model(LogisticRegression(**lr_params),"logistic_regression",lr_params)

best_model = rf_model if f1_rf >= f1_lr else lr_model
best_name  = "Random Forest" if best_model is rf_model else "Logistic Regression"
print(f"✔️  Best model → {best_name}")

# 7. Save Artifacts for Deployment -------------------------------------------
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(best_model, "model_artifacts/model.pkl")
joblib.dump(scaler,     "model_artifacts/scaler.pkl")
joblib.dump(le_product, "model_artifacts/product_encoder.pkl")
joblib.dump(le_region,  "model_artifacts/region_encoder.pkl")
with open("model_artifacts/feature_names.txt","w") as f:
    f.writelines([f"{c}\n" for c in features])
print("✔️  Artifacts saved to ./model_artifacts")

# 8. Quick In-Notebook Prediction --------------------------------------------
sample = pd.DataFrame({
    "price":[299.99],"quantity":[2],"month":[1],"day_of_week":[1],
    "product_enc":[le_product.transform(["Tablet"])],
    "region_enc":[le_region.transform(["North"])]
})
sample_scaled = scaler.transform(sample[features])
proba = best_model.predict_proba(sample_scaled)[1]
pred  = "High Value" if proba >= 0.5 else "Regular"
print(f"Prediction → {pred} (prob={proba:.2%})")

# 9. Next Steps ---------------------------------------------------------------
print(
    "\nOpen MLflow UI in Studio (Compute ▸ MLflow Tracking Servers) "
    "to compare runs, promote the best model, and deploy an endpoint."
)
