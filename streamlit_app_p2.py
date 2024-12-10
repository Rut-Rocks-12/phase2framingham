import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV




# Title and Introduction
st.title("Phase 2 Project: Framingham Study")
st.write("#### By Karthik Gutta (i6306967) & Rutvik Karupothula (i6317004)")
st.write('# Research Question')
st.write("""
         
What are the most significant predictors of hospitalized myocardial infarction (HOSPMI) during follow-up, and how accurately can these predictors classify individuals at risk of an MI using a predictive model?
""")

# Data Loading
DATA_URL = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv'

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

data = load_data(DATA_URL)

# Display Dataset
st.write("# Dataset Preparation")
st.write(data.head())
st.write("Dataset Shape:", data.shape)

# Selecting Relevant Variables
st.write("## Selecting Relevant Variables")
st.write("""
We identified the following variables as most relevant for our analysis to answer the research question:  

- **Demographics:** SEX, AGE  
- **Behavior:** CURSMOKE, CIGPDAY, BMI  
- **Clinical:** TOTCHOL, SYSBP, DIABP, DIABETES, BPMEDS, GLUCOSE, PREVHYP, PREVMI, PREVCHD, PREVAP  
- **Period:** PERIOD  
- **Outcome:** HOSPMI, TIME, TIMEMI, DEATH  

This gives us a total of 20 variables to work with.
""")

selected_columns = [
    'SEX', 'AGE', 'CURSMOKE', 'CIGPDAY', 'BMI', 'TOTCHOL', 'SYSBP', 
    'DIABP', 'DIABETES', 'BPMEDS', 'GLUCOSE', 'PREVHYP', 'PREVMI', 
    'PREVCHD', 'PREVAP', 'PERIOD', 'HOSPMI', 'TIME', 'TIMEMI', 'DEATH'
]
new_data = data[selected_columns]

# Data Exploration
st.write("# Data Exploration and Cleaning")
st.write(new_data.describe())

# Missing Values
st.write("## Missing Values")
missing_values = new_data.isnull().sum()
st.write("""
From the above summary, we can observe missing values in multiple variables.  
We will handle these systematically using appropriate imputation methods.
""")
st.write(missing_values[missing_values > 0])

# Imputation Strategies
st.write("## Imputation Strategies")

#Visualizing the Variables before Imputation

selected_variable = st.selectbox(
    "Select a variable to view its distribution:",
    ["CIGPDAY", "BMI", "TOTCHOL", "BPMEDS", "GLUCOSE"]
)

# Interactive Plot for Selected Variable
st.write(f"### Distribution of {selected_variable}")
if selected_variable == "CIGPDAY":
    st.write("""
    #### CIGPDAY: KNN Imputation  
    All participants are from Framingham, Massachusetts, suggesting local patterns in smoking habits.  
    Therefore, **KNN imputation** is suitable for this variable.
    """)
    fig, ax = plt.subplots()
    ax.hist(new_data['CIGPDAY'], bins=10, color='blue', edgecolor='black')
    ax.set_title('Distribution of CIGPDAY')
    ax.set_xlabel('CIGPDAY (Number of Cigarettes per Day)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

elif selected_variable == "BMI":
    st.write("""
    #### BMI: Mean Imputation  
    The distribution of BMI is approximately normal, making **mean** imputation an appropriate choice.
    """)
    fig, ax = plt.subplots()
    ax.hist(new_data['BMI'], bins=10, color='green', edgecolor='black')
    ax.set_title('Distribution of BMI')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

elif selected_variable == "TOTCHOL":
    st.write("""
    #### TOTCHOL: Mean Imputation  
    A roughly normal distribution is observed, so **mean** imputation is selected for TOTCHOL.
    """)
    fig, ax = plt.subplots()
    ax.hist(new_data['TOTCHOL'], bins=10, color='orange', edgecolor='black')
    ax.set_title('Distribution of TOTCHOL')
    ax.set_xlabel('Total Cholesterol (TOTCHOL)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

elif selected_variable == "BPMEDS":
    st.write("""
    #### BPMEDS: Mode Imputation  
    BPMEDS is categorical. Therefore, **mode** imputation is chosen to handle missing values.
    """)
    fig, ax = plt.subplots()
    ax.hist(new_data['BPMEDS'], bins=5, color='purple', edgecolor='black')
    ax.set_title('Distribution of BPMEDS')
    ax.set_xlabel('BPMEDS (Binary)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

elif selected_variable == "GLUCOSE":
    st.write("""
    #### GLUCOSE: KNN Imputation  
    A non-normal distribution is observed for GLUCOSE. **KNN imputation** is suitable here.
    """)
    fig, ax = plt.subplots()
    ax.hist(new_data['GLUCOSE'], bins=10, color='red', edgecolor='black')
    ax.set_title('Distribution of GLUCOSE')
    ax.set_xlabel('GLUCOSE')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


# CIGPDAY (KNN Imputation)
cigpday_data = new_data[['CIGPDAY']]
knn_imputer = KNNImputer(n_neighbors=5)
cigpday_imputed = knn_imputer.fit_transform(cigpday_data)
new_data.loc[:, 'CIGPDAY'] = cigpday_imputed

# BMI (Mean Imputation)
new_data.loc[:,'BMI'] = new_data['BMI'].fillna(new_data['BMI'].mean())

# TOTCHOL (Mean Imputation)
new_data.loc[:,'TOTCHOL'] = new_data['TOTCHOL'].fillna(new_data['TOTCHOL'].mean())

# BPMEDS (Mode Imputation)
new_data.loc[:, 'BPMEDS'] = new_data['BPMEDS'].fillna(new_data['BPMEDS'].mode()[0])

# GLUCOSE (KNN Imputation)
GLUCOSE_data = new_data[['GLUCOSE']]
knn_imputer = KNNImputer(n_neighbors=5)
GLUCOSE_imputed = knn_imputer.fit_transform(GLUCOSE_data)
new_data.loc[:, 'GLUCOSE'] = GLUCOSE_imputed

st.write('Now that the data has been imputed, we can now consider erroneous Data')


# Erroneous Data
st.write("## Erroneous Data")
st.write("""
Some inconsistencies and physiologically implausible values were detected and corrected:
""")

# Inconsistencies
st.write("#### Inconsistencies in CURSMOKE and CIGPDAY")
st.write('Here we try to detect inconsistency where CURSMOKE is 1 but CIGPDAY is 0 or missing')
inconsistent_smoking = new_data[(new_data['CURSMOKE'] == 1) & (new_data['CIGPDAY'] == 0)]
st.write(inconsistent_smoking)

st.write("#### BPMEDS Usage Without PREVHYP")
st.write('Here we try to check if BPMEDS are being used but PREVHYP or HYPERTEN is 0')
inconsistent_bpmeds = new_data[(new_data['BPMEDS'] == 1) & (new_data['PREVHYP'] == 0)]
st.write(inconsistent_bpmeds)

st.write('Now you can see that, they both return empty datasets, so theres not errors there')

st.write("Physiologically Impossible Values")

# Count of outliers for specified conditions
outliers = {
    "TOTCHOL > 300": new_data[new_data["TOTCHOL"] > 300].shape[0],
    "SYSBP > 180": new_data[new_data["SYSBP"] > 180].shape[0],
    "DIABP > 120": new_data[new_data["DIABP"] > 120].shape[0],
    "DIABP < 40": new_data[new_data["DIABP"] < 40].shape[0],
    "GLUCOSE > 400": new_data[new_data["GLUCOSE"] > 400].shape[0],
    "GLUCOSE < 70": new_data[new_data["GLUCOSE"] < 70].shape[0],
}

# Outliers
st.write("## Outliers")
st.write('Here we look at the number of outliers for the specific variables, where the values were beyond physiological ranges')
for condition, count in outliers.items():
    st.write(f"- **{condition}**: {count}")

st.write("""
These rows were removed to avoid negative effects on the data:  
""")


impossible_values = {
    'TOTCHOL': 300,
    'SYSBP': 180,
    'DIABP': [40, 120],
    'GLUCOSE': [70, 400]
}

new_data.loc[new_data['TOTCHOL'] > impossible_values['TOTCHOL'], 'TOTCHOL'] = np.nan
new_data.loc[new_data['SYSBP'] > impossible_values['SYSBP'], 'SYSBP'] = np.nan
new_data.loc[(new_data['DIABP'] < impossible_values['DIABP'][0]) | (new_data['DIABP'] > impossible_values['DIABP'][1]), 'DIABP'] = np.nan
new_data.loc[(new_data['GLUCOSE'] < impossible_values['GLUCOSE'][0]) | (new_data['GLUCOSE'] > impossible_values['GLUCOSE'][1]), 'GLUCOSE'] = np.nan

new_data = new_data.dropna()
st.write("Shape after handling missing and erroneous values:", new_data.shape)

st.write("Now taking another look at dataset to see if there are any more missing values")
st.write(new_data.isnull().sum())


st.write("""
When checking for the erroneous data we capped the data based on physiological values therefore we do not need to check for outliers in those columns (TOTCHOL, SYSBP, DIABP, and GLUCOSE). 
We also do not need to check for outliers in the categorical data. 

For the numerical variables this leaves us: AGE, BMI, TIMEMI, TIME.
""")
st.write('To check for the outliers in these variables, we used the IQR')
st.write('Below you can select which variables you want to see outliers for')

columns_to_check = ['BMI', 'AGE', 'TIMEMI', 'TIME']

@st.cache_data
def detect_outliers(data, columns):
    outliers = {}
    for column in columns:
        q1, q3 = np.percentile(data[column], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers[column] = data[column][(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

outliers = detect_outliers(new_data, columns_to_check)

selected_outlier_column = st.selectbox(
    "Select a column to view outliers:",
    columns_to_check  # The columns to check for outliers
)

# Display outliers for the selected column
st.write(f"#### Outliers in {selected_outlier_column}")
st.write(outliers[selected_outlier_column])

st.write("""
AGE and TIME show there to be no outliers. However, the outliers in BMI and TIMEMI should not be ignored. These values while statistically categorized as outliers are still medically plausible and clinically relevant.

Both the outliers in TIMEMI & BMI show how individual variability in disease progression and body compisition can influence the timing and risk of a myocardial infarction.
""")

# Final Dataset
st.write("### Final Dataset Shape")
st.write(new_data.shape)


st.write('# Describing and Visualizing the Data')

st.write('## Correlating all variables to HOSPMI')
# Descriptive statistics for each group
yes = new_data[new_data['HOSPMI'] == 1].describe().T
no = new_data[new_data['HOSPMI'] == 0].describe().T
colors = sns.color_palette("coolwarm", as_cmap=True)

# Create subplots for the two categories (hospitalized vs not hospitalized)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Heatmap for hospitalized group
sns.heatmap(
    yes[['mean']],
    annot=True,
    cmap=colors,
    linewidths=0.4,
    linecolor='black',
    cbar=False,
    fmt='.2f',
    ax=axes[0]
)
axes[0].set_title('Hospitalized Due to Myocardial Infarction')

# Heatmap for not hospitalized group
sns.heatmap(
    no[['mean']],
    annot=True,
    cmap=colors,
    linewidths=0.4,
    linecolor='black',
    cbar=False,
    fmt='.2f',
    ax=axes[1]
)
axes[1].set_title('No Hospitalization Due to Myocardial Infarction')

# Adjust layout and show plot in Streamlit
fig.tight_layout(pad=3)
st.pyplot(fig)

# Identify numerical and categorical features
col = list(new_data.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(new_data[i].unique()) > 6:
      if i != "TIMEMI":
        numerical_features.append(i)
      else:
        print(f"Column '{i}' is not included in numerical features.")
    else:
        categorical_features.append(i)

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)


# Inline layout
st.write("## Distribution Exploration of Selected Variables")

# Choose visualization type
visualization_type = st.selectbox(
    "Choose Visualization Type",
    ["Distributions", "Correlations", "Feature vs HOSPMI"],
)

# Visualization options
if visualization_type == "Distributions":
    st.write("### Feature Distributions")
    
    # Dropdown for selecting numerical or categorical features
    feature_type = st.radio("Choose Feature Type", ["Numerical", "Categorical"])
    
    if feature_type == "Numerical":
        feature = st.selectbox("Select a Numerical Feature", numerical_features)
        st.write(f"#### Distribution of {feature}")
        fig, ax = plt.subplots()
        sns.histplot(new_data[feature], bins=20, kde=True, color="blue", edgecolor="black")
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    elif feature_type == "Categorical":
        feature = st.selectbox("Select a Categorical Feature", categorical_features)
        st.write(f"#### Distribution of {feature}")
        fig, ax = plt.subplots()
        sns.countplot(x=feature, data=new_data, palette="coolwarm", edgecolor="black")
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)

elif visualization_type == "Correlations":
    st.write("### Correlation Heatmap")
    correlation_matrix = new_data[numerical_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

elif visualization_type == "Feature vs HOSPMI":
    st.write("### Feature Distributions by HOSPMI")
    
    feature_type = st.radio("Choose Feature Type", ["Numerical", "Categorical"])
    
    if feature_type == "Numerical":
        feature = st.selectbox("Select a Numerical Feature", numerical_features)
        st.write(f"#### {feature} vs HOSPMI")
        fig, ax = plt.subplots()
        sns.histplot(
            data=new_data, x=feature, hue="HOSPMI", kde=True, palette="coolwarm", edgecolor="black"
        )
        ax.set_title(f"{feature} vs HOSPMI")
        st.pyplot(fig)
    elif feature_type == "Categorical":
        feature = st.selectbox("Select a Categorical Feature", categorical_features)
        st.write(f"#### {feature} vs HOSPMI")
        fig, ax = plt.subplots()
        sns.countplot(x=feature, data=new_data, hue="HOSPMI", palette="coolwarm", edgecolor="black")
        ax.set_title(f"{feature} vs HOSPMI")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        st.pyplot(fig)



# Define normal and non-normal feature sets
normal_distri_variables = ["AGE", "BMI", "TOTCHOL", "SYSBP", "DIABP", "GLUCOSE"]
non_normal_distri_variables = ["CIGPDAY", "TIME"]

st.write('# Data Analysis')
# Data Scaling
st.write("## Feature Scaling")
# Correlation with HOSPMI
st.write("### Correlation with HOSPMI")
corr = new_data.corrwith(new_data["HOSPMI"]).sort_values(ascending=False).to_frame()
corr.columns = ["HOSPMI"]
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation with HOSPMI")
st.pyplot(fig)

mms = MinMaxScaler()
for col in non_normal_distri_variables:
    new_data[col] = mms.fit_transform(new_data[[col]])

ss = StandardScaler()
for col in normal_distri_variables:
    new_data[col] = ss.fit_transform(new_data[[col]])
    

st.write("### Scaled Data (Preview)")
st.dataframe(new_data.head())

# ANOVA Test for Numerical Features
st.write("### ANOVA Test for Numerical Features")
features_num = new_data.loc[:,numerical_features]
target = new_data['HOSPMI']

best_features = SelectKBest(score_func = f_classif, k='all')
fit = best_features.fit(features_num, target)

anova_scores = pd.DataFrame(data = fit.scores_,index = list(features_num.columns), columns = ['ANOVA Score'])


fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(anova_scores.sort_values("ANOVA Score", ascending=True), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("ANOVA Scores with HOSPMI")
st.pyplot(fig)


# Chi-Square Test for Categorical Features
st.write("### Chi-Square Test for Categorical Features")
features_cat = new_data.loc[:,categorical_features]
target = new_data['HOSPMI']

best_features = SelectKBest(score_func = chi2, k='all')
fit = best_features.fit(features_cat, target)

chi2_scores = pd.DataFrame(data = fit.scores_,index = list(features_cat.columns), columns = ['Chi Squared Test'])

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(chi2_scores.sort_values(ascending = True, by = 'Chi Squared Test'), annot = True, ax=ax, cmap="coolwarm")
ax.set_title("Chi-Square Scores with HOSPMI")
st.pyplot(fig)

st.write("### Insights")
st.write("- Based on ANOVA scores, we might exclude features with very low scores (TIME).")
st.write("- Based on Chi-Square scores, we might exclude features with very low scores (PERIOD).")

st.write('# Predictive Modelling')

#Defining are Feature and Target variables
X = new_data[['BPMEDS', 'CURSMOKE', 'PREVHYP', 'SEX', 'DIABETES', 'DEATH', 'PREVAP', 'PREVCHD', 'PREVMI', 'BMI', 'DIABP', 'SYSBP', 'AGE', 'GLUCOSE', 'TOTCHOL', 'CIGPDAY']]
y = new_data['HOSPMI']

#Random Forest Model
st.write('## Random Forest Classifier')
# Step 1: Define feature matrix X and target variable y
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 2: Initialize the classifier
clf = RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=100, random_state=0)

# Step 3: Train the classifier
clf.fit(train_X, train_y)

# Step 4: Make predictions
prediction = clf.predict(test_X)

# Step 5: Evaluate the prediction
st.write('Accuracy score:', accuracy_score(y_true=test_y, y_pred=prediction))

# Confusion Matrix
cm = confusion_matrix(y_true=test_y, y_pred=prediction, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

# ROC-AUC Curve
y_pred_prob = clf.predict_proba(test_X)[:, 1]
fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)
roc_auc = roc_auc_score(test_y, y_pred_prob)

# Plot both graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the confusion matrix
disp.plot(ax=axes[0])
axes[0].set_title("Confusion Matrix")

# Plot the ROC-AUC curve
axes[1].plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
axes[1].plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC-AUC Curve")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.5)

st.pyplot(fig)

#KNN Model
st.write('## KNN Classifier')

# Step 1: Split the dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=18)

# Step 2: Initialize the K-Nearest Neighbors Classifier
clf_knn = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='manhattan')

# Step 3: Train the classifier
clf_knn.fit(train_X, train_y)

# Step 4: Make predictions
prediction_knn = clf_knn.predict(test_X)

# Step 5: Evaluate the prediction
st.write('Accuracy score:', accuracy_score(y_true=test_y, y_pred=prediction_knn))

# Confusion Matrix
cm_knn = confusion_matrix(y_true=test_y, y_pred=prediction_knn, normalize='true')
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=clf_knn.classes_)

# ROC-AUC Curve
y_pred_prob_knn = clf_knn.predict_proba(test_X)[:, 1]  # Get predicted probabilities for positive class
fpr_knn, tpr_knn, thresholds_knn = roc_curve(test_y, y_pred_prob_knn)
roc_auc_knn = roc_auc_score(test_y, y_pred_prob_knn)

# Plot both graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the confusion matrix
disp_knn.plot(ax=axes[0])
axes[0].set_title("Confusion Matrix (KNN)")

# Plot the ROC-AUC curve
axes[1].plot(fpr_knn, tpr_knn, label=f"ROC Curve (AUC = {roc_auc_knn:.2f})")
axes[1].plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC-AUC Curve (KNN)")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.5)

st.pyplot(fig)

#Logistic Regression Model
st.write('## Logisitc Regression Classifier')

from sklearn.linear_model import LogisticRegression

# Step 1: Split the dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Step 2: Initialize the Logistic Regression Classifier
clf_logreg = LogisticRegression(max_iter=1000)

# Step 3: Train the classifier
clf_logreg.fit(train_X, train_y)

# Step 4: Make predictions
prediction_logreg = clf_logreg.predict(test_X)

# Step 5: Evaluate the prediction
st.write('Accuracy score:', accuracy_score(y_true=test_y, y_pred=prediction_logreg))

# Confusion Matrix
cm_logreg = confusion_matrix(y_true=test_y, y_pred=prediction_logreg, normalize='true')
disp_logreg = ConfusionMatrixDisplay(confusion_matrix=cm_logreg, display_labels=clf_logreg.classes_)

# ROC-AUC Curve
y_pred_prob_logreg = clf_logreg.predict_proba(test_X)[:, 1]  # Get predicted probabilities for positive class
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(test_y, y_pred_prob_logreg)
roc_auc_logreg = roc_auc_score(test_y, y_pred_prob_logreg)

# Plot both graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the confusion matrix
disp_logreg.plot(ax=axes[0])
axes[0].set_title("Confusion Matrix (Logistic Regression)")

# Plot the ROC-AUC curve
axes[1].plot(fpr_logreg, tpr_logreg, label=f"ROC Curve (AUC = {roc_auc_logreg:.2f})")
axes[1].plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC-AUC Curve (Logistic Regression)")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.5)

st.pyplot(fig)

st.write('# Conclusion')
st.write("""
RQ: What are the most significant predictors of hospitalized myocardial infarction (HOSPMI) during follow-up, and how accurately can these predictors classify individuals at risk of an MI using a predictive model?

To check which predictors across the different categories are most relevant in predicting a myocardal infarction we can look at the correlation with HOSPMI after performing feature engineering. As we can see the PREVMI (0.4), PREVCHD (0.33), DEATH (0.24), PREVAP (0.23), SEX (-0.2) has the hieghest correlation. Although these variables are strong predictors. Other predictors that are less strong include: DIABETES, SYSBP, PREVHYP. The predictors with the least correlation include: BPMEDS (0.071), PERIOD (-0.0071), and TIME (-0.0078).

The final accuracy and ROC AUC scores are the following for the different models;

Random Forest: Accuracy: 0.93 , AUC: 0.82

KNN: Accuracy: 0.92 , AUC: 0.75

Logistic Regression: Accuracy: 0.92, AUC: 0.83
""")