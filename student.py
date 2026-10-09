import warnings
warnings.filterwarnings("ignore")

import joblib
import optuna
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

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
    brier_score_loss,
    mean_absolute_error,
    r2_score
)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# ==========================================================
# LOAD DATASET
# ==========================================================

DATASET = "combined_dataset.csv"

df = pd.read_csv(DATASET)

print("="*70)
print("KNOWLEDGE DISTILLATION USING DECISION TREE")
print("="*70)

print("\nDataset Loaded Successfully\n")

print("Healthy Subjects :", (df["Label"]==0).sum())
print("Apnea Subjects   :", (df["Label"]==1).sum())
print("Total Subjects   :", len(df))

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
print("="*70)
print("DATASET SUMMARY")
print("="*70)

print(f"Training Subjects : {len(X_train)}")
print(f"Testing Subjects  : {len(X_test)}")

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

# ==========================================================
# LOAD TEACHER MODEL
# ==========================================================

teacher = joblib.load(

    "teacher_13features.pkl"

)

print("\nTeacher Model Loaded Successfully")

# ==========================================================
# TEACHER SOFT LABELS
# ==========================================================

teacher_train_prob = teacher.predict_proba(
    X_train
)[:,1]

teacher_test_prob = teacher.predict_proba(
    X_test
)[:,1]

teacher_train_label = (
    teacher_train_prob >= 0.50
).astype(int)

teacher_test_label = (
    teacher_test_prob >= 0.50
).astype(int)

# ==========================================================
# CONFIDENCE WEIGHTS
# ==========================================================

teacher_weights = 1 + 3 * (teacher_train_prob - 0.5) ** 2
teacher_weights = np.clip(teacher_weights, 1.0, 1.75)

# ==========================================================
# METRIC FUNCTION
# ==========================================================

def calculate_metrics(y_true, probability, threshold):

    prediction = (

        probability >= threshold

    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(

        y_true,

        prediction

    ).ravel()

    specificity = tn / (tn + fp)

    return {

        "ROC_AUC": roc_auc_score(

            y_true,

            probability

        ),

        "Accuracy": accuracy_score(

            y_true,

            prediction

        ),

        "Precision": precision_score(

            y_true,

            prediction,

            zero_division=0

        ),

        "Sensitivity": recall_score(

            y_true,

            prediction,

            zero_division=0

        ),

        "Specificity": specificity,

        "F1": f1_score(

            y_true,

            prediction,

            zero_division=0

        ),

        "MCC": matthews_corrcoef(

            y_true,

            prediction

        ),

        "Brier": brier_score_loss(

            y_true,

            probability

        )

    }

# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

def best_threshold(y_true, probability):

    best_t = 0.50

    best_mcc = -1

    for t in np.arange(

        0.20,

        0.81,

        0.01

    ):

        pred = (

            probability >= t

        ).astype(int)

        mcc = matthews_corrcoef(

            y_true,

            pred

        )

        if mcc > best_mcc:

            best_mcc = mcc

            best_t = t

    return best_t

# ==========================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ==========================================================

print("\n")
print("=" * 70)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

def objective(trial):

    params = {

        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy", "log_loss"]
        ),

        "splitter": trial.suggest_categorical(
            "splitter",
            ["best", "random"]
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            20
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10
        ),

        "min_weight_fraction_leaf": trial.suggest_float(
            "min_weight_fraction_leaf",
            0.0,
            0.10
        ),

        "max_features": trial.suggest_categorical(
            "max_features",
            [None, "sqrt", "log2"]
        ),

        "max_leaf_nodes": trial.suggest_int(
            "max_leaf_nodes",
            5,
            60
        ),

        "min_impurity_decrease": trial.suggest_float(
            "min_impurity_decrease",
            0.0,
            0.02
        ),

        "class_weight": trial.suggest_categorical(
            "class_weight",
            [None, "balanced"]
        ),

        "ccp_alpha": trial.suggest_float(
            "ccp_alpha",
            0.0,
            0.02
        ),

        "random_state": 42
    }

    auc_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        teacher_labels = teacher_train_label[train_idx]
        y_val = y_train.iloc[val_idx]

        student = DecisionTreeClassifier(
            **params
        )

        weights = teacher_weights[train_idx]

        student.fit(

           X_tr,

           teacher_labels,

        sample_weight=weights)


        probability = student.predict_proba(
    X_val
)[:,1]

        roc = roc_auc_score(
    y_val,
    probability
)

        mae = mean_absolute_error(
    teacher_train_prob[val_idx],
    probability
)

        r2 = r2_score(
    teacher_train_prob[val_idx],
    probability
)

        score = (

    0.60 * roc

    - 0.20 * mae

    + 0.20 * r2

)

    auc_scores.append(score)
    return float(np.mean(auc_scores))


# ==========================================================
# RUN OPTUNA
# ==========================================================

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=500,
    show_progress_bar=False
)

best_params = study.best_params.copy()

best_params["random_state"] = 42

print("\n")
print("=" * 70)
print("BEST STUDENT PARAMETERS")
print("=" * 70)

for key, value in best_params.items():

    print(f"{key:28s}: {value}")

print("\nBest Validation ROC-AUC : {:.4f}".format(
    study.best_value
))

# ==========================================================
# 5-FOLD VALIDATION
# ==========================================================

validation_results = []

thresholds = []

for train_idx, val_idx in cv.split(X_train, y_train):

    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    teacher_labels = teacher_train_label[train_idx]

    y_val = y_train.iloc[val_idx]

    student = DecisionTreeClassifier(

        **best_params

    )

    weights = teacher_weights[train_idx]

    student.fit(

    X_tr,

    teacher_labels,

    sample_weight=weights

)

    probability = student.predict_proba(

        X_val

    )[:,1]

    threshold = best_threshold(

        y_val,

        probability

    )

    thresholds.append(

        threshold

    )

    metrics = calculate_metrics(

        y_val,

        probability,

        threshold

    )

    validation_results.append(

        metrics

    )

# ==========================================================
# VALIDATION SUMMARY
# ==========================================================

results = pd.DataFrame(

    validation_results

)

FINAL_THRESHOLD = np.mean(

    thresholds

)

print("\n")
print("="*70)
print("VALIDATION SUMMARY")
print("="*70)

print(f"ROC-AUC              : {results['ROC_AUC'].mean():.4f}")
print(f"Accuracy             : {results['Accuracy'].mean():.4f}")
print(f"Precision            : {results['Precision'].mean():.4f}")
print(f"Sensitivity          : {results['Sensitivity'].mean():.4f}")
print(f"Specificity          : {results['Specificity'].mean():.4f}")
print(f"F1 Score             : {results['F1'].mean():.4f}")
print(f"MCC                  : {results['MCC'].mean():.4f}")

print()

print(f"Optimal Threshold    : {FINAL_THRESHOLD:.2f}")

# ==========================================================
# TRAIN FINAL STUDENT
# ==========================================================

student = DecisionTreeClassifier(

    **best_params

)

student.fit(

    X_train,

    teacher_train_label,

    sample_weight=teacher_weights

)

# ==========================================================
# TEST PREDICTION
# ==========================================================
teacher_probability = teacher.predict_proba(

    X_test

)[:,1]

student_probability = student.predict_proba(

    X_test

)[:,1]

student_metrics = calculate_metrics(

    y_test,

    student_probability,

    FINAL_THRESHOLD

)
# ==========================================================
# KNOWLEDGE DISTILLATION METRICS
# ==========================================================


student_mae = mean_absolute_error(

    teacher_probability,

    student_probability

)

student_r2 = r2_score(

    teacher_probability,

    student_probability

)

eps = 1e-7

teacher_clip = np.clip(

    teacher_probability,

    eps,

    1-eps

)

student_clip = np.clip(

    student_probability,

    eps,

    1-eps

)

student_kl = np.mean(

    teacher_clip *

    np.log(

        teacher_clip /

        student_clip

    )

    +

    (1-teacher_clip)

    *

    np.log(

        (1-teacher_clip)

        /

        (1-student_clip)

    )

)

joblib.dump(

    student,

    "student_decision_tree.pkl"

)

print("\n")
print("="*70)
print("STUDENT PERFORMANCE")
print("="*70)

for key in [

    "ROC_AUC",

    "Accuracy",

    "Precision",

    "Sensitivity",

    "Specificity",

    "F1",

    "MCC"

]:

    print(

        f"{key:15s}: {student_metrics[key]:.4f}"

    )

    print()

print(f"MAE            : {student_mae:.4f}")

print(f"R2 Score       : {student_r2:.4f}")

print(f"KL Divergence  : {student_kl:.4f}")

    # ==========================================================
# TEACHER PERFORMANCE
# ==========================================================

teacher_probability = teacher.predict_proba(

    X_test

)[:,1]

teacher_metrics = calculate_metrics(

    y_test,

    teacher_probability,

    FINAL_THRESHOLD

)

print("\n")
print("="*70)
print("TEACHER PERFORMANCE")
print("="*70)

for key in [

    "ROC_AUC",

    "Accuracy",

    "Precision",

    "Sensitivity",

    "Specificity",

    "F1",

    "MCC"

]:

    print(

        f"{key:15s}: {teacher_metrics[key]:.4f}"

    )

# ==========================================================
# TEACHER vs STUDENT COMPARISON
# ==========================================================

comparison = pd.DataFrame({

    "Metric":[

        "ROC-AUC",

        "Accuracy",

        "Precision",

        "Sensitivity",

        "Specificity",

        "F1 Score",

        "MCC"

    ],

    "Teacher":[

        teacher_metrics["ROC_AUC"],

        teacher_metrics["Accuracy"],

        teacher_metrics["Precision"],

        teacher_metrics["Sensitivity"],

        teacher_metrics["Specificity"],

        teacher_metrics["F1"],

        teacher_metrics["MCC"]

    ],

    "Student":[

        student_metrics["ROC_AUC"],

        student_metrics["Accuracy"],

        student_metrics["Precision"],

        student_metrics["Sensitivity"],

        student_metrics["Specificity"],

        student_metrics["F1"],

        student_metrics["MCC"]

    ]

})

print("\n")
print("="*70)
print("TEACHER vs STUDENT")
print("="*70)

print(

    comparison.round(4).to_string(index=False)

)

# ==========================================================
# BEST MODEL
# ==========================================================

if teacher_metrics["ROC_AUC"] >= student_metrics["ROC_AUC"]:

    best_model = "Teacher (XGBoost)"

else:

    best_model = "Student (Decision Tree)"

print("\n")
print("="*70)
print("BEST MODEL")
print("="*70)

print(best_model)

# ==========================================================
# SAVE STUDENT
# ==========================================================

joblib.dump(

    student,

    "student_decision_tree.pkl"

)

print("\nStudent model saved successfully.")

print("student_decision_tree.pkl")

# ==========================================================
# FINAL PROJECT SUMMARY
# ==========================================================

print("\n")
print("="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"Dataset Size       : {len(df)}")

print(f"Healthy Subjects   : {(df['Label']==0).sum()}")

print(f"Apnea Subjects     : {(df['Label']==1).sum()}")

print(f"Training Subjects  : {len(X_train)}")

print(f"Testing Subjects   : {len(X_test)}")

print()

print(f"Teacher ROC-AUC    : {teacher_metrics['ROC_AUC']:.4f}")

print(f"Student ROC-AUC    : {student_metrics['ROC_AUC']:.4f}")

print()

print(f"Teacher Accuracy   : {teacher_metrics['Accuracy']:.4f}")

print(f"Student Accuracy   : {student_metrics['Accuracy']:.4f}")

print()

print("Teacher Model      : teacher_xgboost.pkl")

print("Student Model      : student_decision_tree.pkl")

print("\nKnowledge Distillation Completed Successfully")

print("="*70)