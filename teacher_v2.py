# ==========================================================
# ECG SLEEP APNEA DETECTION
# Teacher Model V2
# XGBoost + Optuna + Threshold Optimization
# ==========================================================

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

df = pd.read_csv("combined_dataset.csv")

print("=" * 75)
print("        ECG SLEEP APNEA DETECTION USING XGBOOST")
print("=" * 75)

print("\nDataset Loaded Successfully\n")

print(df.head())

print("\n")

print("Healthy Subjects :", (df.Label==0).sum())

print("Apnea Subjects   :", (df.Label==1).sum())

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

    stratify=y,

    random_state=42

)

print("\n")

print("="*75)

print("TRAIN TEST SPLIT")

print("="*75)

print("Training Subjects :",len(X_train))

print("Testing Subjects  :",len(X_test))

print("\nTraining Distribution")

print(y_train.value_counts())

print("\nTesting Distribution")

print(y_test.value_counts())

# ==========================================================
# STRATIFIED K FOLD
# ==========================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

# ==========================================================
# METRIC FUNCTION
# ==========================================================

def calculate_metrics(

    y_true,

    probability,

    threshold=0.5

):

    prediction = (

        probability >= threshold

    ).astype(int)

    tn,fp,fn,tp = confusion_matrix(

        y_true,

        prediction

    ).ravel()

    specificity = tn/(tn+fp)

    return {

        "ROC_AUC":

        roc_auc_score(

            y_true,

            probability

        ),

        "Accuracy":

        accuracy_score(

            y_true,

            prediction

        ),

        "Precision":

        precision_score(

            y_true,

            prediction,

            zero_division=0

        ),

        "Sensitivity":

        recall_score(

            y_true,

            prediction,

            zero_division=0

        ),

        "Specificity":

        specificity,

        "F1":

        f1_score(

            y_true,

            prediction,

            zero_division=0

        ),

        "MCC":

        matthews_corrcoef(

            y_true,

            prediction

        ),

        "Brier":

        brier_score_loss(

            y_true,

            probability

        )

    }

# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

def find_best_threshold(

    y_true,

    probability

):

    best_threshold = 0.50

    best_mcc = -1

    for t in np.arange(

        0.20,

        0.81,

        0.01

    ):

        pred = (

            probability>=t

        ).astype(int)

        mcc = matthews_corrcoef(

            y_true,

            pred

        )

        if mcc > best_mcc:

            best_mcc = mcc

            best_threshold = t

    return best_threshold

# ==========================================================
# OPTUNA OBJECTIVE
# ==========================================================

print("\n")
print("="*75)
print("OPTUNA HYPERPARAMETER OPTIMIZATION")
print("="*75)

def objective(trial):

    params = {

        "objective":"binary:logistic",

        "eval_metric":"logloss",

        "tree_method":"hist",

        "random_state":42,

        "n_estimators":trial.suggest_int(
            "n_estimators",
            100,
            600
        ),

        "max_depth":trial.suggest_int(
            "max_depth",
            2,
            5
        ),

        "learning_rate":trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True
        ),

        "subsample":trial.suggest_float(
            "subsample",
            0.70,
            0.90
        ),

        "colsample_bytree":trial.suggest_float(
            "colsample_bytree",
            0.70,
            0.90
        ),

        "gamma":trial.suggest_float(
            "gamma",
            0,
            2
        ),

        "min_child_weight":trial.suggest_int(
            "min_child_weight",
            1,
            5
        ),

        "reg_alpha":trial.suggest_float(
            "reg_alpha",
            0,
            2
        ),

        "reg_lambda":trial.suggest_float(
            "reg_lambda",
            1,
            5
        ),

        "max_delta_step":trial.suggest_int(
            "max_delta_step",
            0,
            5
        )

    }

    auc_scores=[]

    for train_idx,val_idx in cv.split(X_train,y_train):

        X_tr=X_train.iloc[train_idx]
        X_val=X_train.iloc[val_idx]

        y_tr=y_train.iloc[train_idx]
        y_val=y_train.iloc[val_idx]

        model=xgb.XGBClassifier(**params)

        model.fit(
            X_tr,
            y_tr,
            verbose=False
        )

        prob=model.predict_proba(X_val)[:,1]

        auc_scores.append(
            roc_auc_score(
                y_val,
                prob
            )
        )

    return np.mean(auc_scores)

# ==========================================================
# RUN OPTUNA
# ==========================================================

study=optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=100,
    show_progress_bar=True
)

best_params=study.best_params

best_params["objective"]="binary:logistic"
best_params["eval_metric"]="logloss"
best_params["tree_method"]="hist"
best_params["random_state"]=42

print("\n")
print("="*75)
print("OPTUNA FINISHED")
print("="*75)

print("\nBest Validation ROC-AUC : {:.4f}".format(
    study.best_value
))

print("\nBest Parameters\n")

for k,v in best_params.items():

    if k not in [
        "objective",
        "eval_metric",
        "tree_method",
        "random_state"
    ]:

        print(f"{k:20s}: {v}")

# ==========================================================
# VALIDATION
# ==========================================================

print("\n")
print("="*75)
print("5-FOLD VALIDATION")
print("="*75)

validation=[]

best_thresholds=[]

fold=1

for train_idx,val_idx in cv.split(X_train,y_train):

    X_tr=X_train.iloc[train_idx]
    X_val=X_train.iloc[val_idx]

    y_tr=y_train.iloc[train_idx]
    y_val=y_train.iloc[val_idx]

    model=xgb.XGBClassifier(**best_params)

    model.fit(
        X_tr,
        y_tr,
        verbose=False
    )

    train_prob=model.predict_proba(
        X_tr
    )[:,1]

    threshold=find_best_threshold(
        y_tr,
        train_prob
    )

    best_thresholds.append(threshold)

    val_prob=model.predict_proba(
        X_val
    )[:,1]

    metrics=calculate_metrics(
        y_val,
        val_prob,
        threshold
    )

    validation.append(metrics)

    print("\n--------------------------------")

    print(f"Fold {fold}")

    print("--------------------------------")

    print(f"Threshold    : {threshold:.2f}")

    for key,value in metrics.items():

        print(f"{key:15s}: {value:.4f}")

    fold+=1

results=pd.DataFrame(validation)

print("\n")
print("="*75)
print("VALIDATION SUMMARY")
print("="*75)

for col in results.columns:

    print(

        f"{col:15s}: "

        f"{results[col].mean():.4f}"

        f" ± "

        f"{results[col].std():.4f}"

    )

FINAL_THRESHOLD=np.mean(best_thresholds)

print("\n")

print("Optimal Threshold :",round(FINAL_THRESHOLD,2))

# ==========================================================
# TRAIN FINAL TEACHER (ALL FEATURES)
# ==========================================================

print("\n")
print("="*75)
print("TRAINING FINAL TEACHER (13 FEATURES)")
print("="*75)

teacher = xgb.XGBClassifier(**best_params)

teacher.fit(
    X_train,
    y_train,
    verbose=False
)

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
print("="*75)
print("FEATURE IMPORTANCE")
print("="*75)

for i,row in importance.iterrows():

    print(

        f"{i+1:2d}. "

        f"{row['Feature']:12s}"

        f"{row['Importance']:.5f}"

    )

# ==========================================================
# EXTERNAL TEST (13 FEATURES)
# ==========================================================

print("\n")
print("="*75)
print("TEST RESULTS (13 FEATURES)")
print("="*75)

prob13 = teacher.predict_proba(

    X_test

)[:,1]

metrics13 = calculate_metrics(

    y_test,

    prob13,

    FINAL_THRESHOLD

)

for k,v in metrics13.items():

    print(f"{k:15s}: {v:.4f}")

# ==========================================================
# TOP 5 FEATURES
# ==========================================================

TOP5 = importance["Feature"].head(5).tolist()

print("\n")
print("="*75)
print("TOP 5 FEATURES")
print("="*75)

for i,f in enumerate(TOP5,1):

    print(f"{i}. {f}")

# ==========================================================
# TRAIN TOP 5 TEACHER
# ==========================================================

X_train_top5 = X_train[TOP5]

X_test_top5 = X_test[TOP5]

teacher_top5 = xgb.XGBClassifier(**best_params)

teacher_top5.fit(

    X_train_top5,

    y_train,

    verbose=False

)

# ==========================================================
# TEST TOP 5
# ==========================================================

print("\n")
print("="*75)
print("TEST RESULTS (TOP 5)")
print("="*75)

prob5 = teacher_top5.predict_proba(

    X_test_top5

)[:,1]

metrics5 = calculate_metrics(

    y_test,

    prob5,

    FINAL_THRESHOLD

)

for k,v in metrics5.items():

    print(f"{k:15s}: {v:.4f}")

# ==========================================================
# COMPARISON
# ==========================================================

print("\n")
print("="*75)
print("MODEL COMPARISON")
print("="*75)

comparison = pd.DataFrame({

    "Metric":[

        "ROC_AUC",

        "Accuracy",

        "Precision",

        "Sensitivity",

        "Specificity",

        "F1",

        "MCC",

        "Brier"

    ],

    "Teacher_13":[

        metrics13["ROC_AUC"],

        metrics13["Accuracy"],

        metrics13["Precision"],

        metrics13["Sensitivity"],

        metrics13["Specificity"],

        metrics13["F1"],

        metrics13["MCC"],

        metrics13["Brier"]

    ],

    "Teacher_Top5":[

        metrics5["ROC_AUC"],

        metrics5["Accuracy"],

        metrics5["Precision"],

        metrics5["Sensitivity"],

        metrics5["Specificity"],

        metrics5["F1"],

        metrics5["MCC"],

        metrics5["Brier"]

    ]

})

print(comparison.to_string(index=False))

# ==========================================================
# SAVE MODELS
# ==========================================================

joblib.dump(
    teacher,
    "teacher_13features.pkl"
)

joblib.dump(
    teacher_top5,
    "teacher_top5.pkl"
)

# ==========================================================
# FINAL DECISION
# ==========================================================

print("\n")
print("="*75)
print("FINAL MODEL SELECTION")
print("="*75)

if metrics5["ROC_AUC"] > metrics13["ROC_AUC"]:

    FINAL_MODEL = teacher_top5
    FINAL_NAME = "teacher_top5.pkl"
    FINAL_FEATURES = TOP5
    FINAL_METRICS = metrics5

elif metrics5["ROC_AUC"] == metrics13["ROC_AUC"]:

    if metrics5["MCC"] >= metrics13["MCC"]:

        FINAL_MODEL = teacher_top5
        FINAL_NAME = "teacher_top5.pkl"
        FINAL_FEATURES = TOP5
        FINAL_METRICS = metrics5

    else:

        FINAL_MODEL = teacher
        FINAL_NAME = "teacher_13features.pkl"
        FINAL_FEATURES = FEATURES
        FINAL_METRICS = metrics13

else:

    FINAL_MODEL = teacher
    FINAL_NAME = "teacher_13features.pkl"
    FINAL_FEATURES = FEATURES
    FINAL_METRICS = metrics13

joblib.dump(

    FINAL_MODEL,

    "teacher_xgboost.pkl"

)

print("\nBest Teacher Selected")

print(FINAL_NAME)

print("\nSelected Features\n")

for f in FINAL_FEATURES:

    print(f)

# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n")
print("="*75)
print("FINAL TEST PERFORMANCE")
print("="*75)

for k,v in FINAL_METRICS.items():

    print(f"{k:15s}: {v:.4f}")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

if FINAL_NAME=="teacher_top5.pkl":

    probability=teacher_top5.predict_proba(

        X_test_top5

    )[:,1]

else:

    probability=teacher.predict_proba(

        X_test

    )[:,1]

prediction=(

    probability>=FINAL_THRESHOLD

).astype(int)

tn,fp,fn,tp=confusion_matrix(

    y_test,

    prediction

).ravel()

print("\n")
print("="*75)
print("CONFUSION MATRIX")
print("="*75)

print(f"True Positive : {tp}")

print(f"True Negative : {tn}")

print(f"False Positive: {fp}")

print(f"False Negative: {fn}")

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

print("\n")
print("="*75)
print("PROJECT SUMMARY")
print("="*75)

print(f"Dataset Size              : {len(df)}")

print(f"Training Subjects         : {len(X_train)}")

print(f"Testing Subjects          : {len(X_test)}")

print(f"Validation ROC-AUC        : {study.best_value:.4f}")

print(f"Optimal Threshold         : {FINAL_THRESHOLD:.2f}")

print(f"Final Teacher             : {FINAL_NAME}")

print(f"Number of Features        : {len(FINAL_FEATURES)}")

print("\nFeatures Used")

for f in FINAL_FEATURES:

    print(f)

print("\nTeacher Model Saved Successfully")

print("teacher_xgboost.pkl")

print("\n")
print("="*75)
print("END OF TRAINING")
print("="*75)