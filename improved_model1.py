import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Back
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, \
    log_loss, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.base import clone
import logging
import os


# Load the dataset from the specified CSV file
raw_data = pd.read_csv('dataset_B_05_2020.csv')

# Remove the URL column as it is not a feature for prediction
raw_data = raw_data.drop("url", axis=1)

# Replace -1,-2 values in domain_age and domain_registration_length with their median values
domain_age_median_value = round(raw_data.loc[~raw_data['domain_age'].isin([-2, -1]), 'domain_age'].median())
raw_data.loc[~raw_data['domain_age'].isin([-2, -1]), 'domain_age'] = domain_age_median_value
domain_reg_length_median = raw_data.loc[~raw_data['domain_registration_length'].isin([-1]), 'domain_registration_length'].median()
raw_data.loc[~raw_data['domain_registration_length'].isin([-1]), 'domain_registration_length'] = domain_reg_length_median
domain_age_median_value = round(raw_data.loc[~raw_data['domain_age'].isin([-2, -1, 0]), 'domain_age'].median())


# Encode the status column into binary labels (1 for phishing, 0 for legitimate)
raw_data['status'] = raw_data['status'].map({'phishing': 1, 'legitimate': 0})

# Compute the correlation matrix for all features
feature_correlation_matrix = raw_data.corr()

# Extract features with significant correlation to the target variable
target_correlations = feature_correlation_matrix['status'].drop('status').abs().sort_values(ascending=False)
significant_features = target_correlations[target_correlations >= 0.1].index.tolist()

# Identify and handle highly correlated feature pairs to reduce redundancy
high_correlation_pairs = []
for i in range(len(feature_correlation_matrix.columns)):
    for j in range(i + 1, len(feature_correlation_matrix.columns)):
        if abs(feature_correlation_matrix.iloc[i, j]) > 0.8:
            high_correlation_pairs.append((feature_correlation_matrix.columns[i], feature_correlation_matrix.columns[j],
                                           feature_correlation_matrix.iloc[i, j]))

if high_correlation_pairs:
    # Remove one feature from each highly correlated pair, keeping the more relevant one
    features_to_exclude = set()
    for feature1, feature2, _ in high_correlation_pairs:
        if target_correlations[feature1] > target_correlations[feature2]:
            features_to_exclude.add(feature2)
        else:
            features_to_exclude.add(feature1)

# Select the final set of features after removing redundant ones
final_feature_set = [col for col in significant_features if col not in features_to_exclude]
print(f"Selected features after correlation-based extraction: {final_feature_set}")

# Create a preprocessed dataset with selected features and the target
preprocessed_data = raw_data[final_feature_set + ['status']]
'''
# 3.5: Save the preprocessed dataset to a new CSV file
preprocessed_data.to_csv('preprocessed_phishing_data.csv', index=False)
'''
# Prepare the feature matrix and target vector
feature_matrix = preprocessed_data.drop(columns=['status'])
target_vector = preprocessed_data['status']


#grouping values greater than 9 into a single category 10 for the nb_slash feature
feature_matrix['nb_slash'] = feature_matrix['nb_slash'].apply(lambda x: 10 if x > 9 else x)

#grouping values greater than 9 into a single category 10 for the nb_dots feature
#feature_matrix['nb_dots'] = feature_matrix['nb_dots'].apply(lambda x: 10 if x > 9 else x)

#grouping values greater than 4 into a single category 5 for the nb_eq feature
#feature_matrix['nb_eq'] = feature_matrix['nb_eq'].apply(lambda x: 5 if x > 4 else x)

# 4.2: Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42, stratify=target_vector
)
print("Train-Test split complete.")
print(f"train_features shape: {train_features.shape}")
print(f"train_labels shape: {train_labels.shape}")
print(f"test_features shape: {test_features.shape}")
print(f"test_labels shape: {test_labels.shape}")

#num_features = ['ratio_digits_url', 'nb_hyperlinks', 'length_url', 'ratio_intHyperlinks', 'length_hostname', 'ratio_digits_host', 'longest_word_path']

# Step 10: Train and evaluate the default CatBoost model
print("====================================")
print("Training default CatBoost and evaluating:")
categorical_features = [ 'nb_www', 'phish_hints', 'nb_slash', 'nb_eq', 'nb_dots']
cb = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=10,
    eval_metric='Accuracy',
    verbose=100
)
cb.fit(train_features, train_labels, cat_features=categorical_features)
cb_predictions = cb.predict(test_features)

# Evaluate the default CatBoost model
cb_accuracy = accuracy_score(test_labels, cb_predictions)
print(f"Default CatBoost Accuracy: {cb_accuracy:.4f}")
print("Classification Report for Default CatBoost:")
print(classification_report(test_labels, cb_predictions))
print("Confusion Matrix for Default CatBoost:")
print(confusion_matrix(test_labels, cb_predictions))

# Meta-ensemble with two CatBoost models (balanced and unbalanced splits)
# Split 1: Balanced (Stratified)
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42, stratify=target_vector
)

# Split 2: Default (Unbalanced)
X_train_unbal, X_test_unbal, y_train_unbal, y_test_unbal = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42
)

# Train CatBoost on balanced split
cb_bal = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=10,
    eval_metric='Accuracy',
    verbose=0
)
cb_bal.fit(X_train_bal, y_train_bal, cat_features=categorical_features)

# Train CatBoost on unbalanced split
cb_unbal = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=10,
    eval_metric='Accuracy',
    verbose=0
)
cb_unbal.fit(X_train_unbal, y_train_unbal, cat_features=categorical_features)

# Get prediction probabilities for the test set (use the balanced test set for evaluation)
cb_bal_pred = cb_bal.predict_proba(X_test_bal)[:, 1]
cb_unbal_pred = cb_unbal.predict_proba(X_test_bal)[:, 1]

# Stack predictions as meta-features
meta_features = np.column_stack((cb_bal_pred, cb_unbal_pred))

# Train meta-model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(meta_features, y_test_bal)  # Use test labels for meta-model training

# Final predictions
final_pred = meta_model.predict(meta_features)

# Evaluation
meta_accuracy = accuracy_score(y_test_bal, final_pred)
print(f"Meta-ensemble Accuracy: {meta_accuracy:.4f}")
print("Classification Report for Meta-ensemble:")
print(classification_report(y_test_bal, final_pred))
print("Confusion Matrix for Meta-ensemble:")
print(confusion_matrix(y_test_bal, final_pred))


print("====================================")
print("10-Fold Cross Validation for Meta-Ensemble (2 CatBoost models):")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
meta_cv_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(feature_matrix, target_vector), 1):
    # Split for this fold
    X_train, X_val = feature_matrix.iloc[train_idx], feature_matrix.iloc[val_idx]
    y_train, y_val = target_vector.iloc[train_idx], target_vector.iloc[val_idx]

    # Balanced split for CatBoost 1
    X_train_bal, _, y_train_bal, _ = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    # Unbalanced split for CatBoost 2
    X_train_unbal, _, y_train_unbal, _ = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    cb_bal_fold = CatBoostClassifier(
        iterations=700,
        learning_rate=0.03,
        depth=10,
        eval_metric='Accuracy',
        verbose=0
    )
    cb_unbal_fold = CatBoostClassifier(
        iterations=700,
        learning_rate=0.03,
        depth=10,
        eval_metric='Accuracy',
        verbose=0
    )

    cb_bal_fold.fit(X_train_bal, y_train_bal, cat_features=categorical_features)
    cb_unbal_fold.fit(X_train_unbal, y_train_unbal, cat_features=categorical_features)

    cb_bal_pred = cb_bal_fold.predict_proba(X_val)[:, 1]
    cb_unbal_pred = cb_unbal_fold.predict_proba(X_val)[:, 1]

    meta_features_fold = np.column_stack((cb_bal_pred, cb_unbal_pred))

    meta_model_fold = LogisticRegression()
    meta_model_fold.fit(meta_features_fold, y_val)
    final_pred_fold = meta_model_fold.predict(meta_features_fold)

    acc = accuracy_score(y_val, final_pred_fold)
    meta_cv_accuracies.append(acc)
    print(f"Fold {fold} Meta-Ensemble Accuracy: {acc:.4f}")

print(f"Mean Meta-Ensemble CV Accuracy: {np.mean(meta_cv_accuracies):.4f}")
print(f"Std Meta-Ensemble CV Accuracy: {np.std(meta_cv_accuracies):.4f}")

