import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Back
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, \
    log_loss, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
import logging
import os

# Set up logging to track execution and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Load the dataset from the specified CSV file
try:
    raw_data = pd.read_csv('dataset_B_05_2020.csv')
    logger.info("Dataset loaded successfully.")
except FileNotFoundError:
    logger.error("Dataset file 'dataset_B_05_2020.csv' not found.")
    raise
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise


# Step 2: Summarize the dataset with various statistical insights
def analyze_dataset_summary(dataframe, target_column="status"):
    # 2.1: Display the first few rows of the dataset for a quick overview
    print("====================================")
    print("Displaying dataset head:")
    print(dataframe.head())
    logger.info("Displayed dataset head.")

    # 2.2: Show column information (names, data types, and missing values) in a plotly table
    print("====================================")
    print("Displaying column information as plotly table:")
    column_details = pd.DataFrame({
        "Column_Name": dataframe.columns,
        "Data_Type": dataframe.dtypes.astype(str).values,
        "Missing_Values": dataframe.isnull().sum().values
    })
    plot_table = go.Figure(data=[go.Table(
        header=dict(values=list(column_details.columns), fill_color='lightblue', align='left'),
        cells=dict(values=[column_details[col] for col in column_details.columns],
                   fill_color='white', align='left'))
    ])
    plot_table.update_layout(title="Column Info", height=600)
    plot_table.show()
    logger.info("Displayed column information table.")
    print("Column information output complete.")

    # 2.3: Check for and display any duplicate rows in the dataset
    print("====================================")
    print("Checking for duplicate rows:")
    duplicate_rows = dataframe[dataframe.duplicated()]
    if not duplicate_rows.empty:
        duplicate_table = go.Figure(data=[go.Table(
            header=dict(values=list(duplicate_rows.columns), fill_color='lightblue', align='left'),
            cells=dict(values=[duplicate_rows[col].head(20) for col in duplicate_rows.columns],
                       fill_color='white', align='left'))
        ])
        duplicate_table.update_layout(title="🔍 Duplicate Rows (First 20)", height=600)
        duplicate_table.show()
        logger.info("Displayed duplicate rows.")
    else:
        print("No duplicate rows found.")
        logger.info("No duplicate rows found.")

    # 2.4: Display the distribution of classes in the target column
    print("====================================")
    print("Displaying class distribution:")
    if target_column and target_column in dataframe.columns:
        class_frequencies = dataframe[target_column].value_counts().reset_index()
        class_frequencies.columns = [target_column, "Count"]

        class_dist_plot = go.Figure(data=[go.Table(
            header=dict(values=list(class_frequencies.columns), fill_color='lightblue', align='left'),
            cells=dict(values=[class_frequencies[col] for col in class_frequencies.columns],
                       fill_color='white', align='left'))
        ])
        class_dist_plot.update_layout(title=f"Class Distribution: {target_column}", height=600)
        class_dist_plot.show()
        logger.info("Displayed class distribution.")
        print("Class distribution output complete.")


analyze_dataset_summary(raw_data)

# Step 3: Preprocess the dataset for model training
# 3.1: Remove the URL column as it is not a feature for prediction
raw_data = raw_data.drop("url", axis=1)
logger.info("Removed URL column.")

# 3.2: Replace -1 values in domain_age and domain_registration_length with their median values
print("====================================")
print("Replacing -1 values in domain_age and domain_registration_length with their median")
domain_age_median_value = raw_data.loc[raw_data['domain_age'] != -1, 'domain_age'].median()
raw_data.loc[raw_data['domain_age'] == -1, 'domain_age'] = domain_age_median_value
print(f"Replaced -1 in 'domain_age' with median: {domain_age_median_value}")
domain_reg_length_median = raw_data.loc[
    raw_data['domain_registration_length'] != -1, 'domain_registration_length'].median()
raw_data.loc[raw_data['domain_registration_length'] == -1, 'domain_registration_length'] = domain_reg_length_median
print(f"Replaced -1 in 'domain_registration_length' with median: {domain_reg_length_median}")
logger.info("Replaced -1 values with medians.")

# 3.3: Encode the status column into binary labels (1 for phishing, 0 for legitimate)
print("====================================")
print("Encoding the class label: status")
raw_data['status'] = raw_data['status'].map({'phishing': 1, 'legitimate': 0})
print("Class label 'status' encoded. Mapping: {'phishing': 1, 'legitimate': 0}")
logger.info("Encoded status column.")

# 3.4: Perform correlation analysis and feature extraction
print("====================================")
print("Performing correlation analysis and feature extraction:")
# Compute the correlation matrix for all features
feature_correlation_matrix = raw_data.corr()
logger.info("Computed correlation matrix.")

# Visualize the correlation matrix as a heatmap
print("Displaying correlation matrix as a heatmap:")
correlation_heatmap = px.imshow(
    feature_correlation_matrix,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    title="Correlation Matrix Heatmap",
    width=1000,
    height=1000
)
correlation_heatmap.update_layout(
    xaxis_title="Features",
    yaxis_title="Features",
    font=dict(size=10)
)
correlation_heatmap.show()
logger.info("Displayed correlation heatmap.")
print("Correlation matrix heatmap output complete.")

# Extract features with significant correlation to the target variable
print("====================================")
print("Extracting features based on correlation with target 'status':")
target_correlations = feature_correlation_matrix['status'].drop('status').abs().sort_values(ascending=False)
significant_features = target_correlations[target_correlations >= 0.3].index.tolist()
print(f"Features with absolute correlation >= 0.3 with 'status': {significant_features}")
logger.info(f"Extracted {len(significant_features)} significant features.")

# Identify and handle highly correlated feature pairs to reduce redundancy
print("====================================")
print("Identifying highly correlated feature pairs (correlation > 0.8):")
high_correlation_pairs = []
for i in range(len(feature_correlation_matrix.columns)):
    for j in range(i + 1, len(feature_correlation_matrix.columns)):
        if abs(feature_correlation_matrix.iloc[i, j]) > 0.8:
            high_correlation_pairs.append((feature_correlation_matrix.columns[i], feature_correlation_matrix.columns[j],
                                           feature_correlation_matrix.iloc[i, j]))

if high_correlation_pairs:
    print("Highly correlated feature pairs (correlation > 0.8):")
    for pair in high_correlation_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
    logger.info(f"Found {len(high_correlation_pairs)} highly correlated pairs.")

    # Remove one feature from each highly correlated pair, keeping the more relevant one
    features_to_exclude = set()
    for feature1, feature2, _ in high_correlation_pairs:
        if target_correlations[feature1] > target_correlations[feature2]:
            features_to_exclude.add(feature2)
        else:
            features_to_exclude.add(feature1)
    print(f"Features to remove due to high correlation: {list(features_to_exclude)}")
    logger.info(f"Excluded {len(features_to_exclude)} features due to high correlation.")
else:
    print("No feature pairs with correlation > 0.8 found.")
    logger.info("No highly correlated pairs found.")

# Select the final set of features after removing redundant ones
final_feature_set = [col for col in significant_features if col not in features_to_exclude]
print(f"Selected features after correlation-based extraction: {final_feature_set}")
logger.info(f"Final feature set size: {len(final_feature_set)}.")

# Create a preprocessed dataset with selected features and the target
preprocessed_data = raw_data[final_feature_set + ['status']]
print(f"Preprocessed dataset shape after feature extraction: {preprocessed_data.shape}")
logger.info(f"Preprocessed dataset shape: {preprocessed_data.shape}")

# 3.5: Save the preprocessed dataset to a new CSV file
print("====================================")
print("Saving preprocessed dataset:")
preprocessed_data.to_csv('preprocessed_phishing_data.csv', index=False)
print("Preprocessed dataset saved as 'preprocessed_phishing_data.csv'")
logger.info("Saved preprocessed dataset to CSV.")

# Step 4: Apply SMOTE and perform train-test split
print("====================================")
print("Applying SMOTE (if needed) and performing train-test split:")
# Prepare the feature matrix and target vector
feature_matrix = preprocessed_data.drop(columns=['status'])
target_vector = preprocessed_data['status']
logger.info("Prepared feature matrix and target vector.")

'''
# 4.1: Apply SMOTE to balance the dataset before splitting (currently commented out)
print("Applying SMOTE on entire dataset:")
# Calculate class distribution before SMOTE
class_distribution_before = target_vector.value_counts().reset_index()
class_distribution_before.columns = ['status', 'Count']
class_distribution_before['status'] = class_distribution_before['status'].map({1: 'phishing', 0: 'legitimate'})

# Apply SMOTE to oversample the minority class
smote_balancer = SMOTE(random_state=42)
balanced_features, balanced_targets = smote_balancer.fit_resample(feature_matrix, target_vector)

# Calculate class distribution after SMOTE
class_distribution_after = balanced_targets.value_counts().reset_index()
class_distribution_after.columns = ['status', 'Count']
class_distribution_after['status'] = class_distribution_after['status'].map({1: 'phishing', 0: 'legitimate'})

# Display class distribution before and after SMOTE in a table
print("Displaying class distribution for entire dataset (before and after SMOTE):")
combined_distributions = pd.DataFrame({
    'Dataset': ['Entire Dataset (Before SMOTE)', 'Entire Dataset (Before SMOTE)',
                'Entire Dataset (After SMOTE)', 'Entire Dataset (After SMOTE)'],
    'Status': (class_distribution_before['status'].tolist() +
               class_distribution_after['status'].tolist()),
    'Count': (class_distribution_before['Count'].tolist() +
              class_distribution_after['Count'].tolist())
})

dist_plot = go.Figure(data=[go.Table(
    header=dict(values=['Dataset', 'Status', 'Count'], fill_color='lightblue', align='left'),
    cells=dict(
        values=[combined_distributions['Dataset'], combined_distributions['Status'], combined_distributions['Count']],
        fill_color='white', align='left'))
])
dist_plot.update_layout(title="Class Distribution for Entire Dataset (Before and After SMOTE)", height=400)
dist_plot.show()
print("Class distribution for entire dataset displayed.")
print(f"Balanced_features shape (entire dataset): {balanced_features.shape}")
print(f"Balanced_targets shape (entire dataset): {balanced_targets.shape}")
logger.info("SMOTE applied and class distribution displayed (commented section).")
'''

# 4.2: Split the dataset into training and testing sets
print("Performing train-test split on preprocessed dataset:")
train_features, test_features, train_labels, test_labels = train_test_split(feature_matrix, target_vector,
                                                                            test_size=0.2, random_state=42)
print("Train-Test split complete.")
print(f"train_features shape: {train_features.shape}")
print(f"train_labels shape: {train_labels.shape}")
print(f"test_features shape: {test_features.shape}")
print(f"test_labels shape: {test_labels.shape}")
logger.info("Completed train-test split.")

# Step 5: Scale the features for consistent model input
print("====================================")
print("Scaling features using StandardScaler:")
feature_scaler = StandardScaler()
train_scaled_features = feature_scaler.fit_transform(train_features)
test_scaled_features = feature_scaler.transform(test_features)
print("Feature scaling complete.")
logger.info("Features scaled successfully.")

# Step 6: Train and evaluate the default SVM model
print("====================================")
print("Training default SVM and evaluating:")
svm_default_model = SVC(random_state=42)
svm_default_model.fit(train_scaled_features, train_labels)
svm_default_predictions = svm_default_model.predict(test_scaled_features)
logger.info("Trained and evaluated default SVM model.")

# Evaluate the default SVM model
svm_default_accuracy = accuracy_score(test_labels, svm_default_predictions)
print(f"Default SVM Accuracy: {svm_default_accuracy:.4f}")
print("Classification Report for Default SVM:")
print(classification_report(test_labels, svm_default_predictions))
print("Confusion Matrix for Default SVM:")
print(confusion_matrix(test_labels, svm_default_predictions))
logger.info(f"Default SVM Accuracy: {svm_default_accuracy:.4f}")

# Step 7: Perform Grid Search to optimize SVM hyperparameters
print("====================================")
print("Performing Grid Search for SVM hyperparameters:")
svm_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
    'degree': [2, 3, 5]  # Relevant only for poly kernel
}
svm_grid_search = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(train_scaled_features, train_labels)
logger.info("Performed Grid Search for SVM hyperparameters.")

# Extract and evaluate the best SVM model
print("Best parameters found by Grid Search:")
print(svm_grid_search.best_params_)
best_svm_model = svm_grid_search.best_estimator_
svm_best_predictions = best_svm_model.predict(test_scaled_features)
logger.info(f"Best SVM parameters: {svm_grid_search.best_params_}")

svm_best_accuracy = accuracy_score(test_labels, svm_best_predictions)
print(f"Best SVM Accuracy: {svm_best_accuracy:.4f}")
print("Classification Report for Best SVM:")
print(classification_report(test_labels, svm_best_predictions))
print("Confusion Matrix for Best SVM:")
print(confusion_matrix(test_labels, svm_best_predictions))
logger.info(f"Best SVM Accuracy: {svm_best_accuracy:.4f}")

# Step 8: Train and evaluate the default RandomForest model
print("====================================")
print("Training default RandomForest and evaluating:")
rf_default_model = RandomForestClassifier(random_state=42)
rf_default_model.fit(train_features, train_labels)
rf_default_predictions = rf_default_model.predict(test_features)
logger.info("Trained and evaluated default RandomForest model.")

# Evaluate the default RandomForest model
rf_default_accuracy = accuracy_score(test_labels, rf_default_predictions)
print(f"Default RandomForest Accuracy: {rf_default_accuracy:.4f}")
print("Classification Report for Default RandomForest:")
print(classification_report(test_labels, rf_default_predictions))
print("Confusion Matrix for Default RandomForest:")
print(confusion_matrix(test_labels, rf_default_predictions))
logger.info(f"Default RandomForest Accuracy: {rf_default_accuracy:.4f}")

# Step 9: Perform Grid Search to optimize RandomForest hyperparameters
print("====================================")
print("Performing Grid Search for RandomForest hyperparameters:")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(train_features, train_labels)
logger.info("Performed Grid Search for RandomForest hyperparameters.")

# Extract and evaluate the best RandomForest model
print("Best parameters found by Grid Search:")
print(rf_grid_search.best_params_)
best_rf_model = rf_grid_search.best_estimator_
rf_best_predictions = best_rf_model.predict(test_features)
logger.info(f"Best RandomForest parameters: {rf_grid_search.best_params_}")

rf_best_accuracy = accuracy_score(test_labels, rf_best_predictions)
print(f"Best Random Forest Accuracy: {rf_best_accuracy:.4f}")
print("Classification Report for Best Random Forest:")
print(classification_report(test_labels, rf_best_predictions))
print("Confusion Matrix for Best Random Forest:")
print(confusion_matrix(test_labels, rf_best_predictions))
logger.info(f"Best RandomForest Accuracy: {rf_best_accuracy:.4f}")

# Step 10: Train and evaluate the default CatBoost model
print("====================================")
print("Training default CatBoost and evaluating:")
categorical_features = ['google_index', 'nb_www', 'domain_in_title', 'phish_hints', 'ip']
cb_default_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    verbose=100
)
cb_default_model.fit(train_features, train_labels, cat_features=categorical_features)
cb_default_predictions = cb_default_model.predict(test_features)
logger.info("Trained and evaluated default CatBoost model.")

# Evaluate the default CatBoost model
cb_default_accuracy = accuracy_score(test_labels, cb_default_predictions)
print(f"Default CatBoost Accuracy: {cb_default_accuracy:.4f}")
print("Classification Report for Default CatBoost:")
print(classification_report(test_labels, cb_default_predictions))
print("Confusion Matrix for Default CatBoost:")
print(confusion_matrix(test_labels, cb_default_predictions))
logger.info(f"Default CatBoost Accuracy: {cb_default_accuracy:.4f}")

# Step 11: Perform Grid Search to optimize CatBoost hyperparameters
print("====================================")
print("Performing Grid Search for CatBoost hyperparameters:")
cb_param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'iterations': [100, 300]
}
cb_grid_search = GridSearchCV(CatBoostClassifier(verbose=0), cb_param_grid, cv=3, scoring='accuracy')
cb_grid_search.fit(train_features, train_labels, cat_features=categorical_features)
logger.info("Performed Grid Search for CatBoost hyperparameters.")

# Extract and evaluate the best CatBoost model
print("Best parameters found by Grid Search:")
print(cb_grid_search.best_params_)
best_cb_model = cb_grid_search.best_estimator_
cb_best_predictions = best_cb_model.predict(test_features)
logger.info(f"Best CatBoost parameters: {cb_grid_search.best_params_}")

cb_best_accuracy = accuracy_score(test_labels, cb_best_predictions)
print(f"Best CatBoost Accuracy: {cb_best_accuracy:.4f}")
print("Classification Report for Best CatBoost:")
print(classification_report(test_labels, cb_best_predictions))
print("Confusion Matrix for Best CatBoost:")
print(confusion_matrix(test_labels, cb_best_predictions))
logger.info(f"Best CatBoost Accuracy: {cb_best_accuracy:.4f}")

# Step 12: Conduct a comparative analysis of the best models
print("====================================")
print("Performing Comparative Analysis:")
logger.info("Starting comparative analysis.")

# Collect accuracy results from the best models
model_accuracies = {
    'Support_Vector_Machine': svm_best_accuracy,
    'Random_Forest': rf_best_accuracy,
    'CatBoost': cb_best_accuracy
}

# Display accuracy results in a simple table format
print("\nModel Accuracies:")
for model_name, accuracy_value in model_accuracies.items():
    print(f"{model_name}: {accuracy_value:.4f}")
logger.info("Displayed model accuracies.")

# Display confusion matrices for each best model
print("\nConfusion Matrices:")
print("Support Vector Machine:")
print(confusion_matrix(test_labels, svm_best_predictions))
print("Random Forest:")
print(confusion_matrix(test_labels, rf_best_predictions))
print("CatBoost:")
print(confusion_matrix(test_labels, cb_best_predictions))
logger.info("Displayed confusion matrices.")

# Display detailed classification reports for each best model
print("\nClassification Reports:")
print("Support Vector Machine:")
print(classification_report(test_labels, svm_best_predictions))
print("Random Forest:")
print(classification_report(test_labels, rf_best_predictions))
print("CatBoost:")
print(classification_report(test_labels, cb_best_predictions))
logger.info("Displayed classification reports.")

# Visualization - Accuracy Comparison (Fig. 5)
accuracy_dataframe = pd.DataFrame(list(model_accuracies.items()), columns=['Model_Name', 'Accuracy_Score'])
accuracy_plot = px.bar(accuracy_dataframe, x='Model_Name', y='Accuracy_Score', title='Comparative Analysis of Accuracy',
                       text='Accuracy_Score')
accuracy_plot.update_traces(texttemplate='%{text:.3f}', textposition='outside')
accuracy_plot.show()
logger.info("Displayed accuracy comparison plot (Fig. 5).")

# Visualization - Confusion Matrix Heatmaps (corresponding to Fig. 2, 3, 4 in the article)
for model_name, predictions in zip(['Support Vector Machine', 'Random Forest', 'CatBoost'],
                                   [svm_best_predictions, rf_best_predictions, cb_best_predictions]):
    confusion_mat = confusion_matrix(test_labels, predictions)
    heatmap_plot = px.imshow(confusion_mat, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                             title=f'{model_name} Confusion Matrix Heatmap')
    heatmap_plot.update_xaxes(title_text='Predicted_Label')
    heatmap_plot.update_yaxes(title_text='True_Label')
    heatmap_plot.show()
    logger.info(f"Displayed confusion matrix heatmap for {model_name} (Fig. 2, 3, or 4).")

# Extract performance metrics for Class-0 (legitimate) and Class-1 (phishing)
model_evaluations = [('Support Vector Machine', svm_best_predictions), ('Random Forest', rf_best_predictions),
                     ('CatBoost', cb_best_predictions)]

legitimate_metrics_df = pd.DataFrame(columns=['Model_Name', 'Precision_Score', 'Recall_Score', 'F1_Score'])
phishing_metrics_df = pd.DataFrame(columns=['Model_Name', 'Precision_Score', 'Recall_Score', 'F1_Score'])

for model_name, predictions in model_evaluations:
    precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(test_labels, predictions)
    legitimate_metrics_df = pd.concat([legitimate_metrics_df, pd.DataFrame(
        {'Model_Name': [model_name], 'Precision_Score': [precision_scores[0]], 'Recall_Score': [recall_scores[0]],
         'F1_Score': [f1_scores[0]]})], ignore_index=True)
    phishing_metrics_df = pd.concat([phishing_metrics_df, pd.DataFrame(
        {'Model_Name': [model_name], 'Precision_Score': [precision_scores[1]], 'Recall_Score': [recall_scores[1]],
         'F1_Score': [f1_scores[1]]})], ignore_index=True)
    logger.info(f"Extracted metrics for {model_name}.")

# Visualization - Comparative Analysis for Class-0 (Fig. 6)
legitimate_melted_data = legitimate_metrics_df.melt(id_vars='Model_Name', var_name='Metric', value_name='Score_Value')
legitimate_comparison_plot = px.bar(legitimate_melted_data, x='Metric', y='Score_Value', color='Model_Name',
                                    barmode='group',
                                    title='Comparative Analysis for Class-0', text='Score_Value')
legitimate_comparison_plot.update_traces(texttemplate='%{text:.4f}', textposition='outside')
legitimate_comparison_plot.show()
logger.info("Displayed comparative analysis for Class-0 (Fig. 6).")

# Visualization - Comparative Analysis for Class-1 (Fig. 7)
phishing_melted_data = phishing_metrics_df.melt(id_vars='Model_Name', var_name='Metric', value_name='Score_Value')
phishing_comparison_plot = px.bar(phishing_melted_data, x='Metric', y='Score_Value', color='Model_Name',
                                  barmode='group',
                                  title='Comparative Analysis for Class-1', text='Score_Value')
phishing_comparison_plot.update_traces(texttemplate='%{text:.4f}', textposition='outside')
phishing_comparison_plot.show()
logger.info("Displayed comparative analysis for Class-1 (Fig. 7).")

# Compute Log Loss for each model (requires probabilities)
# Refit SVM with probability=True to enable log loss calculation
print("====================================")
print("Refitting SVM with probability=True for log loss calculation...")
best_svm_params = svm_grid_search.best_params_
best_svm_model_proba = SVC(**best_svm_params, probability=True, random_state=42)
best_svm_model_proba.fit(train_scaled_features, train_labels)
svm_probability_predictions = best_svm_model_proba.predict_proba(test_scaled_features)[:, 1]
logger.info("Refitted SVM with probability for log loss calculation.")

# Calculate probability predictions for Random Forest
rf_probability_predictions = best_rf_model.predict_proba(test_features)[:, 1]
logger.info("Calculated probability predictions for Random Forest.")

# Calculate probability predictions for CatBoost
cb_probability_predictions = best_cb_model.predict_proba(test_features)[:, 1]
logger.info("Calculated probability predictions for CatBoost.")

# Compute log loss values for all models
log_loss_values = {
    'Support_Vector_Machine': log_loss(test_labels, svm_probability_predictions),
    'Random_Forest': log_loss(test_labels, rf_probability_predictions),
    'CatBoost': log_loss(test_labels, cb_probability_predictions)
}
logger.info("Computed log loss values for all models.")

# Visualization - Comparative Analysis of Log Loss for Class-1 (Fig. 8)
log_loss_dataframe = pd.DataFrame(list(log_loss_values.items()), columns=['Model_Name', 'Log_Loss_Value'])
log_loss_plot = px.bar(log_loss_dataframe, x='Model_Name', y='Log_Loss_Value',
                       title='Comparative Analysis of Log Loss for Class-1', text='Log_Loss_Value')
log_loss_plot.update_traces(texttemplate='%{text:.5f}', textposition='outside')
log_loss_plot.show()
logger.info("Displayed log loss comparison plot (Fig. 8).")

# Step 13: Generate Figure 9 - ROC Comparison
print("====================================")
print("Generating Figure 9 - ROC Comparison:")
logger.info("Generating Figure 9 - ROC Comparison.")

# Compute ROC curves and AUC for each model
fpr_svm, tpr_svm, _ = roc_curve(test_labels, svm_probability_predictions)
fpr_rf, tpr_rf, _ = roc_curve(test_labels, rf_probability_predictions)
fpr_cb, tpr_cb, _ = roc_curve(test_labels, cb_probability_predictions)
auc_svm = roc_auc_score(test_labels, svm_probability_predictions)
auc_rf = roc_auc_score(test_labels, rf_probability_predictions)
auc_cb = roc_auc_score(test_labels, cb_probability_predictions)
logger.info("Computed ROC curves and AUC scores.")

# Create ROC comparison plot
fig9 = go.Figure()
fig9.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm, mode='lines', name=f'SVM (AUC = {auc_svm:.4f}'))
fig9.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {auc_rf:.4f}'))
fig9.add_trace(go.Scatter(x=fpr_cb, y=tpr_cb, mode='lines', name=f'CatBoost (AUC = {auc_cb:.4f}'))
fig9.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
fig9.update_layout(
    title='ROC Comparison of Phishing Detection Models',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    legend_title='Models',
    template='plotly_white'
)
fig9.show()
# # Save to HTML file as fallback
# output_file = 'figure9_roc_comparison.html'
# fig9.write_html(output_file)
# print(f"Figure 9 saved as {output_file}. Open in a web browser to view.")
# logger.info(f"Saved Figure 9 to {output_file}.")
