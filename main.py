import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Back
from sklearn.preprocessing import StandardScaler, LabelEncoder

# [Step 0: Import the dataset]

df = pd.read_csv('dataset_B_05_2020.csv')

# [Step 1: Display dataset summary]

def summarize_dataset(df, class_column="status"):

    #1.1 Display df.head()
    print("====================================")
    print("Displaying dataset head:")
    print(df.head())

    #1.2 Display Column info
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

    #1.3 Check Duplicate rows
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

    #1.4 Check class distribution
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

# [Step 2: Preprocessing]

#2.1 Dropping the URL column
df = df.drop("url", axis=1)

#2.2 replacing -1 values in domain_age and domain_registration_length
print("====================================")
print("Replacing -1 values in domain_age and domain_regitstration_length with their median")

domain_age_median = df.loc[df['domain_age'] != -1, 'domain_age'].median()
df.loc[df['domain_age'] == -1, 'domain_age'] = domain_age_median
print(f"Replaced -1 in 'domain_age' with median: {domain_age_median}")
domain_reg_len_median = df.loc[df['domain_registration_length'] != -1, 'domain_registration_length'].median()
df.loc[df['domain_registration_length'] == -1, 'domain_registration_length'] = domain_reg_len_median
print(f"Replaced -1 in 'domain_registration_length' with median: {domain_reg_len_median}")

#2.3 Class label encoding
print("====================================")
print("Encoding the class label: status")

df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})
print("Class label 'status' encoded. Mapping: {'phishing': 1, 'legitimate': 0}")