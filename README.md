# machine-learning-models-for-Fraud-detection
## Contents

Exploratory Data Analysis (EDA)

Head/shape/describe

Distribution plots (Amount boxplot, Time histogram, full histogram grid)

Class balance check (Class value counts)

Preprocessing & Imbalance Handling

StandardScaler (and a Normalizer demo)

SMOTE oversampling within an imblearn pipeline

### Models

RandomForestClassifier

XGBClassifier (XGBoost) with scale_pos_weight

CatBoostClassifier with scale_pos_weight

Simple hybrid ensemble by averaging RF & XGB probabilities

### Evaluation

Train/validation splits with train_test_split (stratified)

5-fold StratifiedKFold cross-validation

Metrics: Accuracy, Recall, Precision, F1, ROC-AUC

Confusion matrix & classification report

ROC and Precision–Recall curves (per-model and comparison)

Probability comparison scatterplot (RF vs XGB)

### Data

The notebook expects the Kaggle-style credit card fraud dataset schema:

Features: Time, Amount, and anonymized components (V1…V28)

Target: Class (0 = legitimate, 1 = fraud)
