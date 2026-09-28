"""
model.py
Shared code for the Employee Retention & Pay Check app.

Two models trained on the IBM HR Analytics Employee Attrition dataset (Kaggle, 1,470 employees):
  1. Logistic regression  -> probability that an employee will leave (Attrition = Yes/No)
  2. Linear regression    -> expected monthly income for a role and experience profile

Both use a fixed 75/25 train/test split (random_state=42) so every number in the app
can be reproduced.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "employee_attrition.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# Logistic regression (attrition). YearsAtCompany and JobLevel were dropped because they are
# highly correlated with YearsWithCurrManager and MonthlyIncome and flipped coefficient signs,
# which would confuse managers reading the explanation.
LOGIT_NUM = [
    "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked", "TotalWorkingYears",
    "YearsWithCurrManager", "YearsSinceLastPromotion", "TrainingTimesLastYear",
    "StockOptionLevel", "EnvironmentSatisfaction", "JobSatisfaction",
    "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement",
]
LOGIT_CAT = ["OverTime", "BusinessTravel", "MaritalStatus", "Department"]

# Linear regression (monthly income). Department is left out because each job role sits
# inside one department, so it adds no information.
LINEAR_NUM = ["JobLevel", "TotalWorkingYears", "YearsAtCompany", "Education", "PerformanceRating"]
LINEAR_CAT = ["JobRole"]

# Plain-language labels shown to managers
LABELS = {
    "Age": "Age",
    "DistanceFromHome": "Distance from home (km)",
    "MonthlyIncome": "Monthly income",
    "NumCompaniesWorked": "Number of past employers",
    "TotalWorkingYears": "Total work experience (years)",
    "YearsWithCurrManager": "Years with current manager",
    "YearsSinceLastPromotion": "Years since last promotion",
    "TrainingTimesLastYear": "Trainings attended last year",
    "StockOptionLevel": "Stock option level (0-3)",
    "EnvironmentSatisfaction": "Work environment satisfaction",
    "JobSatisfaction": "Job satisfaction",
    "RelationshipSatisfaction": "Relationship with colleagues",
    "WorkLifeBalance": "Work-life balance",
    "JobInvolvement": "Job involvement",
    "OverTime_Yes": "Works overtime",
    "BusinessTravel_Travel_Frequently": "Travels frequently for work",
    "BusinessTravel_Travel_Rarely": "Travels rarely for work",
    "MaritalStatus_Single": "Single",
    "MaritalStatus_Married": "Married",
    "Department_Research & Development": "R&D department",
    "Department_Sales": "Sales department",
    "JobLevel": "Job level (1-5)",
    "YearsAtCompany": "Years at this company",
    "Education": "Education level (1-5)",
    "PerformanceRating": "Performance rating",
}

SCALE_1_4 = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very high"}
WLB_1_4 = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
EDUCATION = {1: "1 - Below college", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}


# ---------------------------------------------------------------------------
# Data and training
# ---------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    return df


def build_logistic() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), LOGIT_NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), LOGIT_CAT),
    ])
    # class_weight="balanced" because only 16% of employees left. Without it the model
    # catches only about 1 in 3 leavers.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("prep", pre), ("model", clf)])


def build_linear() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", LINEAR_NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), LINEAR_CAT),
    ])
    return Pipeline([("prep", pre), ("model", LinearRegression())])


def train_models(df: pd.DataFrame) -> dict:
    """Train both models on 75% of the data and evaluate on the held-out 25%."""
    # Attrition model
    X = df[LOGIT_NUM + LOGIT_CAT]
    y = df["AttritionFlag"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    logit = build_logistic().fit(Xtr, ytr)
    p_test = logit.predict_proba(Xte)[:, 1]

    # Pay model
    Xl = df[LINEAR_NUM + LINEAR_CAT]
    yl = df["MonthlyIncome"]
    Xltr, Xlte, yltr, ylte = train_test_split(Xl, yl, test_size=0.25, random_state=RANDOM_STATE)
    linear = build_linear().fit(Xltr, yltr)
    yl_pred = linear.predict(Xlte)
    resid_sd = float(np.std(yltr - linear.predict(Xltr), ddof=len(LINEAR_NUM) + 9))

    return {
        "logit": logit, "logit_test": (Xte, yte, p_test),
        "linear": linear, "linear_test": (Xlte, ylte, yl_pred),
        "resid_sd": resid_sd,
        "base_rate": float(y.mean()),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def logit_metrics(y_true, p, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    fpr, tpr, _ = roc_curve(y_true, p)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "ROC AUC": roc_auc_score(y_true, p),
        "Recall (leavers caught)": recall_score(y_true, pred),
        "Precision (flags that were right)": precision_score(y_true, pred, zero_division=0),
        "F1 score": f1_score(y_true, pred),
        "cm": (tn, fp, fn, tp), "roc": (fpr, tpr),
    }


def linear_metrics(y_true, y_pred) -> dict:
    return {
        "R squared": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true))),
    }


def coef_table(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = np.ravel(pipe.named_steps["model"].coef_)
    clean = [n.split("__", 1)[1] for n in names]
    out = pd.DataFrame({"feature": clean, "coefficient": coefs})
    out["label"] = out["feature"].map(lambda f: LABELS.get(f, f.replace("JobRole_", "Role: ")))
    return out


# ---------------------------------------------------------------------------
# Explanations for one employee
# ---------------------------------------------------------------------------
def explain_logit(pipe: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    """
    Contribution of each feature to this employee's risk score (log-odds scale).
    Numeric: coefficient x standardised value (0 = company average).
    Categorical: coefficient if the employee is in that category, else 0
    (baseline categories: OverTime=No, Non-Travel, Divorced, Human Resources).
    """
    prep = pipe.named_steps["prep"]
    x = prep.transform(row)
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    tbl = coef_table(pipe)
    tbl["contribution"] = tbl["coefficient"].values * x
    tbl = tbl[tbl["contribution"].abs() > 1e-9]
    return tbl.sort_values("contribution", key=np.abs, ascending=False)


def what_if(pipe: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    """Risk after one change a manager can act on."""
    base = float(pipe.predict_proba(row)[:, 1][0])
    r = row.iloc[0]
    scenarios = []

    def add(name, **changes):
        new = row.copy()
        for k, v in changes.items():
            new[k] = v
        p = float(pipe.predict_proba(new)[:, 1][0])
        scenarios.append({"Action": name, "Risk now": base, "Risk after": p, "Change": p - base})

    if r["OverTime"] == "Yes":
        add("Remove regular overtime", OverTime="No")
    if r["YearsSinceLastPromotion"] >= 2:
        add("Promote this year", YearsSinceLastPromotion=0)
    if r["BusinessTravel"] == "Travel_Frequently":
        add("Cut travel to occasional", BusinessTravel="Travel_Rarely")
    if r["StockOptionLevel"] < 3:
        add("Raise stock option level by 1", StockOptionLevel=int(r["StockOptionLevel"]) + 1)
    if r["JobSatisfaction"] < 4:
        add("Raise job satisfaction by 1 point", JobSatisfaction=int(r["JobSatisfaction"]) + 1)
    if r["EnvironmentSatisfaction"] < 4:
        add("Raise environment satisfaction by 1 point", EnvironmentSatisfaction=int(r["EnvironmentSatisfaction"]) + 1)
    add("Give a 10% pay rise", MonthlyIncome=float(r["MonthlyIncome"]) * 1.10)

    return pd.DataFrame(scenarios).sort_values("Change")


def risk_band(p: float) -> str:
    if p < 0.35:
        return "Low"
    if p < 0.65:
        return "Medium"
    return "High"
