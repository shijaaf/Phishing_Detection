import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore, Back
from sklearn.preprocessing import StandardScaler

# [Step 0: Import the dataset]

df = pd.read_csv('dataset_B_05_2020.csv')

# [Step 1: Display dataset summary]

def summarize_dataset(df, class_column="status"):

    #1.1 Display df.head()
    print(Back.GREEN + "Displaying dataset head:" + Back.RESET)
    print(df.head())

    #1.2 Display Column info
    print(Back.GREEN + "Displaying column information as plotly table:" + Back.RESET)
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
    print(Back.GREEN + "Checking for duplicate rows:" + Back.RESET)
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
        print(Back.GREEN + "Displaying class distribution:" + Back.RESET)
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

#2.1 Find categorical features
categorical_cols = [col for col in df.select_dtypes(include='int').columns
                 if df[col].nunique() <= 5]
print(Back.GREEN + "Columns with Categorical Variables:" + Back.RESET)
print(categorical_cols)

#2.2 One-hot encode categorical features
print(df[categorical_cols].shape)
df_cat = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#2.3 Find numerical features
numerical_cols = [col for col in df.columns
                      if col not in categorical_cols
                      and df[col].dtype in ['int64', 'float64']]
print(Back.GREEN + "Columns with Numerical Variables:" + Back.RESET)
print(numerical_cols)

#2.4 Scale numerical features
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

#2.5 Combine numerical and categorical features
df_final = pd.concat([df_cat, df_num], axis=1)
df_final['status'] = df['status'].reset_index(drop=True)
print(Back.GREEN + "Label encoding and standard scaling operations finished." + Back.RESET)