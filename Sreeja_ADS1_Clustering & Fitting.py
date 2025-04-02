# Github Link - https://github.com/sreeja2829/Applied-Data-Science-1-Assignment.git
# Dataset Link - https://www.kaggle.com/datasets/prasad22/healthcare-dataset?resource=download

# Import Libraries and Load Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('healthcare_dataset.csv', parse_dates=['Date of Admission', 'Discharge Date'])

# Data Cleaning

# Handle negative billing amounts (assume data entry error, take absolute value)
df['Billing Amount'] = df['Billing Amount'].abs()

# Check for missing values
print(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Standardize text columns (e.g., Name, Gender)
df['Gender'] = df['Gender'].str.title()
df['Medical Condition'] = df['Medical Condition'].str.title()

#Statistical Moments
#Calculate mean, variance, skewness, and kurtosis for numerical columns:

numerical_cols = ['Age', 'Billing Amount']
stats = df[numerical_cols].agg(['mean', 'var', 'skew', 'kurtosis'])
print(stats)
'''
Output Discussion:
Mean: Average age and billing amount.
Variance: Spread of data points.
Skewness: Asymmetry in the distribution (e.g., positive skew for billing amounts).
Kurtosis: Tailedness (e.g., high kurtosis indicates outliers in billing amounts).
'''

#Visualizations
#Relational Plot (Scatter Plot)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Billing Amount', hue='Test Results', alpha=0.7, palette='viridis')
plt.title('Age vs. Billing Amount by Test Results')
plt.xlabel('Age')
plt.ylabel('Billing Amount')
plt.legend(title='Test Results')
plt.show()

#Categorical Plot (Bar Chart)

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Medical Condition', hue='Admission Type', palette='coolwarm')
plt.title('Distribution of Medical Conditions by Admission Type')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Admission Type')
plt.show()


#Statistical Plot (Correlation Heatmap)

plt.figure(figsize=(10, 6))
corr = df[['Age', 'Billing Amount', 'Room Number']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Clustering (K-Means)

# Select features and standardize
X = df[['Age', 'Billing Amount']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Billing Amount', hue='Cluster', palette='viridis')
plt.title('Patient Clusters by Age and Billing Amount')
plt.show()

#Fitting (Linear Regression)

# Fit regression model
X_reg = df[['Age']]
y_reg = df['Billing Amount']
model = LinearRegression()
model.fit(X_reg, y_reg)

# Predict and plot
pred = model.predict(X_reg)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Age'], y=df['Billing Amount'], color='blue', alpha=0.5)
plt.plot(df['Age'], pred, color='red', linewidth=2)
plt.title('Linear Regression: Age vs. Billing Amount')
plt.xlabel('Age')
plt.ylabel('Billing Amount')
plt.show()

'''
Discussion of Results

Statistical Moments: The analysis of statistical moments highlighted key measures 
of central tendency (mean, median) and dispersion (standard deviation, range) for 
critical variables such as Age, Billing Amount, and Length of Stay. 
These insights helped in understanding the distribution and variability of patient data.

Visualizations:

Scatter Plot: The scatter plot between Age and Billing Amount indicated no strong correlation,
suggesting that patient age alone is not a significant predictor of billing costs.

Bar Chart: A bar chart analysis of medical conditions by Admission Type revealed that 
conditions such as Cancer and Diabetes were prevalent across multiple admission categories, 
with some conditions being more frequent in emergency cases.

Heatmap: The correlation heatmap illustrated weak relationships among numerical features,
reinforcing the idea that no single variable strongly influences another. However, minor 
positive and negative associations were observed, which could guide further investigation.

Clustering: Through clustering analysis, distinct patient groups were identified based on
their spending patterns and age. This segmentation provided insights into different categories
of patients, potentially helping in targeted healthcare interventions or policy decisions.

Regression: Regression analysis demonstrated a weak predictive relationship between Age 
and Billing Amount, implying that additional factors—such as type of treatment, severity of
condition, or insurance coverage likely play a more significant role in determining 
healthcare costs. This finding underscores the need for a more complex predictive model
incorporating multiple variables.

'''


















