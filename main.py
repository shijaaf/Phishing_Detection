import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Back
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# [Step 1: Import the dataset]
df = pd.read_csv('dataset_B_05_2020.csv')

# [Step 2: Display dataset summary]
def summarize_dataset(df, class_column="status"):
    # 2.1 Display df.head()
    print("====================================")
    print("Displaying dataset head:")
    print(df.head())

    # 2.2 Display Column info
    print("====================================")
    print("Displaying column information as plotly table:")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(col_info.columns), fill_color='lightblue', align='left'),
        cells=dict(values=[col_info[col] for col in col_info.columns],
                   fill_color='white', align='left'))
    ])
    fig.update_layout(title="Column Info", height=600)
    fig.show()
    print("Column information output complete.")

    # 2.3 Check Duplicate rows
    print("====================================")
    print("Checking for duplicate rows:")
    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        fig2 = go.Figure(data=[go.Table(
            header=dict(values=list(duplicates.columns), fill_color='lightblue', align='left'),
            cells=dict(values=[duplicates[col].head(20) for col in duplicates.columns],
                       fill_color='white', align='left'))
        ])
        fig2.update_layout(title="🔍 Duplicate Rows (First 20)", height=600)
        fig2.show()
    else:
        print("No duplicate rows found.")

    # 2.4 Check class distribution
    print("====================================")
    print("Displaying class distribution:")
    if class_column and class_column in df.columns:
        class_counts = df[class_column].value_counts().reset_index()
        class_counts.columns = [class_column, "Count"]

        fig4 = go.Figure(data=[go.Table(
            header=dict(values=list(class_counts.columns), fill_color='lightblue', align='left'),
            cells=dict(values=[class_counts[col] for col in class_counts.columns],
                       fill_color='white', align='left'))
        ])
        fig4.update_layout(title=f"Class Distribution: {class_column}", height=600)
        fig4.show()
        print("Class distribution output complete.")

summarize_dataset(df)

# [Step 3: Preprocessing]
# 3.1 Dropping the URL column
df = df.drop("url", axis=1)

# 3.2 Replacing -1 values in domain_age and domain_registration_length
print("====================================")
print("Replacing -1 values in domain_age and domain_regitstration_length with their median")
domain_age_median = df.loc[df['domain_age'] != -1, 'domain_age'].median()
df.loc[df['domain_age'] == -1, 'domain_age'] = domain_age_median
print(f"Replaced -1 in 'domain_age' with median: {domain_age_median}")
domain_reg_len_median = df.loc[df['domain_registration_length'] != -1, 'domain_registration_length'].median()
df.loc[df['domain_registration_length'] == -1, 'domain_registration_length'] = domain_reg_len_median
print(f"Replaced -1 in 'domain_registration_length' with median: {domain_reg_len_median}")

# 3.3 Class label encoding
print("====================================")
print("Encoding the class label: status")
df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})
print("Class label 'status' encoded. Mapping: {'phishing': 1, 'legitimate': 0}")

# 3.4 Correlation analysis and feature extraction
print("====================================")
print("Performing correlation analysis and feature extraction:")
# Compute correlation matrix
corr_matrix = df.corr()

# Visualize correlation matrix as a heatmap
print("Displaying correlation matrix as a heatmap:")
fig_corr = px.imshow(
    corr_matrix,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    title="Correlation Matrix Heatmap",
    width=1000,
    height=1000
)
fig_corr.update_layout(
    xaxis_title="Features",
    yaxis_title="Features",
    font=dict(size=10)
)
fig_corr.show()
print("Correlation matrix heatmap output complete.")

# Feature extraction based on correlation with target
print("====================================")
print("Extracting features based on correlation with target 'status':")
target_corr = corr_matrix['status'].drop('status').abs().sort_values(ascending=False)
high_corr_features = target_corr[target_corr >= 0.3].index.tolist()
print(f"Features with absolute correlation >= 0.3 with 'status': {high_corr_features}")

# Identify highly correlated feature pairs to reduce redundancy
print("====================================")
print("Identifying highly correlated feature pairs (correlation > 0.8):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("Highly correlated feature pairs (correlation > 0.8):")
    for pair in high_corr_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")

    # Remove one feature from each highly correlated pair
    features_to_remove = set()
    for feat1, feat2, _ in high_corr_pairs:
        # Keep the feature with higher correlation to target
        if target_corr[feat1] > target_corr[feat2]:
            features_to_remove.add(feat2)
        else:
            features_to_remove.add(feat1)
    print(f"Features to remove due to high correlation: {list(features_to_remove)}")
else:
    print("No feature pairs with correlation > 0.8 found.")

# Select final features
selected_features = [col for col in high_corr_features if col not in features_to_remove]
print(f"Selected features after correlation-based extraction: {selected_features}")

# Create preprocessed dataset with selected features
df_preprocessed = df[selected_features + ['status']]
print(f"Preprocessed dataset shape after feature extraction: {df_preprocessed.shape}")

# 3.5 Save preprocessed dataset
print("====================================")
print("Saving preprocessed dataset:")
df_preprocessed.to_csv('preprocessed_dataset.csv', index=False)
print("Preprocessed dataset saved as 'preprocessed_dataset.csv'.")

# [Step 4: SMOTE Application and Train-Test Split]
print("====================================")
print("Applying SMOTE (entire dataset) and performing train-test split:")
# Prepare features and target
X = df_preprocessed.drop(columns=['status'])
y = df_preprocessed['status']
'''
# 4.1 SMOTE Before Train-Test Split (Entire Dataset)
print("Applying SMOTE on entire dataset:")
# Class distribution before SMOTE
class_counts_before = y.value_counts().reset_index()
class_counts_before.columns = ['status', 'Count']
class_counts_before['status'] = class_counts_before['status'].map({1: 'phishing', 0: 'legitimate'})

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Class distribution after SMOTE
class_counts_after = y_resampled.value_counts().reset_index()
class_counts_after.columns = ['status', 'Count']
class_counts_after['status'] = class_counts_after['status'].map({1: 'phishing', 0: 'legitimate'})

# Create first table for entire dataset
print("Displaying class distribution for entire dataset (before and after SMOTE):")
combined_counts_entire = pd.DataFrame({
    'Dataset': ['Entire Dataset (Before SMOTE)', 'Entire Dataset (Before SMOTE)',
                'Entire Dataset (After SMOTE)', 'Entire Dataset (After SMOTE)'],
    'Status': (class_counts_before['status'].tolist() +
               class_counts_after['status'].tolist()),
    'Count': (class_counts_before['Count'].tolist() +
              class_counts_after['Count'].tolist())
})

fig_entire = go.Figure(data=[go.Table(
    header=dict(values=['Dataset', 'Status', 'Count'], fill_color='lightblue', align='left'),
    cells=dict(
        values=[combined_counts_entire['Dataset'], combined_counts_entire['Status'], combined_counts_entire['Count']],
        fill_color='white', align='left'))
])
fig_entire.update_layout(title="Class Distribution for Entire Dataset (Before and After SMOTE)", height=400)
fig_entire.show()
print("Class distribution for entire dataset displayed.")
print(f"X_resampled shape (entire dataset): {X_resampled.shape}")
print(f"y_resampled shape (entire dataset): {y_resampled.shape}")
'''
# 4.2 Train-Test Split
print("Performing train-test split on preprocessed dataset:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-Test split complete.")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# [Step 5: Feature Scaling]
print("====================================")
print("Scaling features using StandardScaler:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

# [Step 6: Train Default SVM and Evaluate]
print("====================================")
print("Training default SVM and evaluating:")
svm_default = SVC(random_state=42)
svm_default.fit(X_train_scaled, y_train)
svm_y_pred_default = svm_default.predict(X_test_scaled)

# Evaluation
svm_accuracy_default = accuracy_score(y_test, svm_y_pred_default)
print(f"Default SVM Accuracy: {svm_accuracy_default:.4f}")
print("Classification Report for Default SVM:")
print(classification_report(y_test, svm_y_pred_default))
print("Confusion Matrix for Default SVM:")
print(confusion_matrix(y_test, svm_y_pred_default))

# [Step 7: Grid Search for SVM Hyperparameters]
print("====================================")
print("Performing Grid Search for SVM hyperparameters:")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
    'degree': [2, 3, 5]  # Relevant only for poly kernel
}
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters and evaluation
print("Best parameters found by Grid Search:")
print(grid_search.best_params_)
best_svm = grid_search.best_estimator_
svm_y_pred_best = best_svm.predict(X_test_scaled)

svm_accuracy_best = accuracy_score(y_test, svm_y_pred_best)
print(f"Best SVM Accuracy: {svm_accuracy_best:.4f}")
print("Classification Report for Best SVM:")
print(classification_report(y_test, svm_y_pred_best))
print("Confusion Matrix for Best SVM:")
print(confusion_matrix(y_test, svm_y_pred_best))

# [Step 8: Train Default RandomForest and Evaluate]
print("====================================")
print("Training default RandomForest and evaluating:")
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
rf_y_pred_default = rf_default.predict(X_test)

# Evaluation
rf_accuracy_default = accuracy_score(y_test, rf_y_pred_default)
print(f"Default RandomForest Accuracy: {rf_accuracy_default:.4f}")
print("Classification Report for Default RandomForest:")
print(classification_report(y_test, rf_y_pred_default))
print("Confusion Matrix for Default RandomForest:")
print(confusion_matrix(y_test, rf_y_pred_default))

# [Step 9: Grid Search for RandomForest Hyperparameters]
print("====================================")
print("Performing Grid Search for RandomForest hyperparameters:")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and evaluation
print("Best parameters found by Grid Search:")
print(grid_search.best_params_)
best_rf = grid_search.best_estimator_
rf_y_pred_best = best_rf.predict(X_test)

rf_accuracy_best = accuracy_score(y_test, rf_y_pred_best)
print(f"Best SVM Accuracy: {rf_accuracy_best:.4f}")
print("Classification Report for Best SVM:")
print(classification_report(y_test, rf_y_pred_best))
print("Confusion Matrix for Best SVM:")
print(confusion_matrix(y_test, rf_y_pred_best))

# [Step 10: Train Default CatBoost and Evaluate]
