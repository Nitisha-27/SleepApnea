import warnings
warnings.filterwarnings("ignore")

import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss
)

# ==========================================================
# LOAD DATASET
# ==========================================================

DATASET = "combined_dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 70)
print("      ECG SLEEP APNEA DETECTION USING XGBOOST")
print("=" * 70)

print("\nDataset Loaded Successfully\n")

print("Healthy :", (df.Label == 0).sum())
print("Apnea   :", (df.Label == 1).sum())
print("Total   :", len(df))

# ==========================================================
# FEATURES
# ==========================================================

FEATURES = [

    "RR",
    "HRV",
    "HeartRate",
    "P",
    "T",
    "PQ",
    "PR",
    "QRS",
    "QT",
    "ST",
    "RT",
    "PT",
    "TPTE"

]

X = df[FEATURES]

y = df["Label"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y

)

print("\n")
print("=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training :", len(X_train))
print("Testing  :", len(X_test))

print("\nTraining Distribution")

print("Healthy :", sum(y_train == 0))
print("Apnea   :", sum(y_train == 1))

print("\nTesting Distribution")

print("Healthy :", sum(y_test == 0))
print("Apnea   :", sum(y_test == 1))

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

# ==========================================================
# METRIC FUNCTION
# ==========================================================

def calculate_metrics(y_true, prob):

    pred = (prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred
    ).ravel()

    specificity = tn / (tn + fp)

    return {

        "ROC_AUC": roc_auc_score(
            y_true,
            prob
        ),

        "Accuracy": accuracy_score(
            y_true,
            pred
        ),

        "Precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Sensitivity": recall_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Specificity": specificity,

        "F1": f1_score(
            y_true,
            pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_true,
            pred
        ),

        "Brier": brier_score_loss(
            y_true,
            prob
        )

    }
# ==========================================================
# OPTUNA OBJECTIVE FUNCTION
# ==========================================================

print("\n")
print("=" * 70)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

def objective(trial):

    params = {

        "objective": "binary:logistic",

        "eval_metric": "logloss",

        "tree_method": "hist",

        "random_state": 42,

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            1000
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.00
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0,
            5
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0,
            5
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0,
            5
        ),

        "max_delta_step": trial.suggest_int(
            "max_delta_step",
            0,
            10
        )

    }

    fold_scores = []

    for train_idx, valid_idx in cv.split(X_train, y_train):

        X_tr = X_train.iloc[train_idx]

        X_val = X_train.iloc[valid_idx]

        y_tr = y_train.iloc[train_idx]

        y_val = y_train.iloc[valid_idx]

        model = xgb.XGBClassifier(
            **params
        )

        model.fit(
            X_tr,
            y_tr,
            verbose=False
        )

        probability = model.predict_proba(
            X_val
        )[:,1]

        score = roc_auc_score(
            y_val,
            probability
        )

        fold_scores.append(score)

    return np.mean(fold_scores)

# ==========================================================
# RUN OPTUNA
# ==========================================================

study = optuna.create_study(

    direction="maximize"

)

study.optimize(

    objective,

    n_trials=100,

    show_progress_bar=True

)

print("\n")
print("=" * 70)
print("OPTUNA FINISHED")
print("=" * 70)

print("\nBest ROC-AUC")

print("{:.4f}".format(

    study.best_value

))

print("\nBest Parameters\n")

for key,value in study.best_params.items():

    print(f"{key:20s} : {value}")

best_params = study.best_params.copy()

best_params["objective"] = "binary:logistic"

best_params["eval_metric"] = "logloss"

best_params["tree_method"] = "hist"

best_params["random_state"] = 42

# ==========================================================
# TRAIN BEST TEACHER MODEL (5-FOLD CV)
# ==========================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

fold_results = []

fold = 1

for train_idx, valid_idx in cv.split(X_train, y_train):

    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[valid_idx]

    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[valid_idx]

    teacher = xgb.XGBClassifier(**best_params)

    teacher.fit(
        X_tr,
        y_tr,
        verbose=False
    )

    probability = teacher.predict_proba(X_val)[:,1]

    metrics = calculate_metrics(
        y_val,
        probability
    )

    fold_results.append(metrics)

    print("\n------------------------------------------")
    print(f"Fold {fold}")
    print("------------------------------------------")

    for key,value in metrics.items():

        print(f"{key:15s}: {value:.4f}")

    fold += 1

# ==========================================================
# MEAN & STD
# ==========================================================

results = pd.DataFrame(fold_results)

print("\n")
print("=" * 70)
print("AVERAGE PERFORMANCE (5-FOLD)")
print("=" * 70)

for col in results.columns:

    print(
        f"{col:15s}: "
        f"{results[col].mean():.4f} ± "
        f"{results[col].std():.4f}"
    )

# ==========================================================
# TRAIN FINAL MODEL
# ==========================================================

print("\n")
print("=" * 70)
print("TRAINING FINAL TEACHER MODEL")
print("=" * 70)

teacher = xgb.XGBClassifier(**best_params)

teacher.fit(
    X_train,
    y_train,
    verbose=False
)

print("Done.")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": teacher.feature_importances_

})

importance = importance.sort_values(

    "Importance",

    ascending=False

).reset_index(drop=True)

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

for i,row in importance.iterrows():

    print(
        f"{i+1:2d}. "
        f"{row['Feature']:12s}"
        f"{row['Importance']:.5f}"
    )

# ==========================================================
# TOP 10 FEATURE SELECTION
# ==========================================================

print("\n")
print("=" * 70)
print("TOP 10 FEATURE SELECTION")
print("=" * 70)

top10_features = importance["Feature"].head(10).tolist()

print("\nSelected Features\n")

for i, feature in enumerate(top10_features, 1):
    print(f"{i}. {feature}")

# ==========================================================
# RETRAIN USING TOP 10 FEATURES
# ==========================================================

X_train_top10 = X_train[top10_features]
X_test_top10 = X_test[top10_features]

teacher_top10 = xgb.XGBClassifier(**best_params)

teacher_top10.fit(
    X_train_top10,
    y_train,
    verbose=False
)

# ==========================================================
# EXTERNAL TEST EVALUATION
# ==========================================================

print("\n")
print("=" * 70)
print("EXTERNAL TEST RESULTS")
print("=" * 70)

prob = teacher_top10.predict_proba(X_test_top10)[:, 1]

metrics = calculate_metrics(
    y_test,
    prob
)

for key, value in metrics.items():
    print(f"{key:15s}: {value:.4f}")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

prediction = (prob >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    prediction
).ravel()

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(f"True Positive : {tp}")
print(f"True Negative : {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    teacher_top10,
    "teacher_xgboost.pkl"
)

print("\n")
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("teacher_xgboost.pkl")

# ==========================================================
# FEATURE IMPORTANCE (TOP 10)
# ==========================================================

print("\n")
print("=" * 70)
print("TOP 10 FEATURES")
print("=" * 70)

for i, row in importance.head(10).iterrows():

    print(
        f"{i+1:2d}. "
        f"{row['Feature']:12s}"
        f"{row['Importance']:.5f}"
    )

# ==========================================================
# AUTOMATIC BEST TOP-K FEATURE SEARCH
# ==========================================================

print("\n")
print("=" * 70)
print("AUTOMATIC TOP-K FEATURE SEARCH")
print("=" * 70)

best_auc = -1

best_k = None

best_features = None

best_model = None

comparison = []

for k in range(3, 11):

    features = importance["Feature"].head(k).tolist()

    X_train_k = X_train[features]

    X_test_k = X_test[features]

    model = xgb.XGBClassifier(**best_params)

    model.fit(
        X_train_k,
        y_train,
        verbose=False
    )

    probability = model.predict_proba(
        X_test_k
    )[:,1]

    metrics = calculate_metrics(
        y_test,
        probability
    )

    comparison.append({

        "TopK": k,

        **metrics

    })

    print(f"\nTop {k}")

    print(f"ROC-AUC : {metrics['ROC_AUC']:.4f}")

    print(f"Accuracy: {metrics['Accuracy']:.4f}")

    if metrics["ROC_AUC"] > best_auc:

        best_auc = metrics["ROC_AUC"]

        best_k = k

        best_features = features

        best_model = model

# ==========================================================
# FINAL BEST MODEL
# ==========================================================

print("\n")
print("=" * 70)
print("BEST FEATURE SUBSET")
print("=" * 70)

print(f"Top {best_k} Features Selected\n")

for i, f in enumerate(best_features, 1):

    print(f"{i}. {f}")

print("\nBest ROC-AUC :", round(best_auc,4))

# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(

    best_model,

    "teacher_xgboost.pkl"

)

print("\nBest Teacher Saved")

print("teacher_xgboost.pkl")

# ==========================================================
# COMPARISON TABLE
# ==========================================================

comparison = pd.DataFrame(comparison)

print("\n")
print("=" * 70)
print("TOP-K COMPARISON")
print("=" * 70)

print(

    comparison.to_string(index=False)

)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("FINAL TEACHER SUMMARY")
print("=" * 70)

print(f"Training Subjects : {len(X_train)}")

print(f"Testing Subjects  : {len(X_test)}")

print(f"Best ROC-AUC      : {best_auc:.4f}")

print(f"Best Top Features : {best_k}")

print("\nSelected Features\n")

for f in best_features:

    print(f)

print("\nTeacher Model Saved Successfully")
