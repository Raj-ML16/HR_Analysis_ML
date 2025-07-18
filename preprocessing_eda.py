import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load datasets
employee_df = pd.read_csv('employee_data_realistic.csv')
business_df = pd.read_csv('business_data.csv')
financial_df = pd.read_csv('financial_data.csv')

print("="*60)
print("DATA PREPROCESSING")
print("="*60)

# 1. Basic Info
print(f"Employee Data Shape: {employee_df.shape}")
print(f"Business Data Shape: {business_df.shape}")
print(f"Financial Data Shape: {financial_df.shape}")

# 2. Check Missing Values
print("\nMissing Values in Employee Data:")
missing_emp = employee_df.isnull().sum()
print(missing_emp[missing_emp > 0])

print("\nMissing Values in Business Data:")
missing_bus = business_df.isnull().sum()
print(missing_bus[missing_bus > 0])

print("\nMissing Values in Financial Data:")
missing_fin = financial_df.isnull().sum()
print(missing_fin[missing_fin > 0])

# 3. Handle Missing Values
# Fill Manager_ID nulls for top-level employees
employee_df['Manager_ID'].fillna('None', inplace=True)

# Fill Resignation_Date and Lead_Time nulls for active employees
employee_df['Resignation_Date'].fillna('Active', inplace=True)
employee_df['Lead_Time'].fillna(0, inplace=True)

print("\nAfter handling missing values:")
print(f"Employee missing values: {employee_df.isnull().sum().sum()}")

# 4. Data Types Conversion
employee_df['Joining_Date'] = pd.to_datetime(employee_df['Joining_Date'])
employee_df['Resignation_Date'] = employee_df['Resignation_Date'].replace('Active', pd.NaT)
employee_df['Resignation_Date'] = pd.to_datetime(employee_df['Resignation_Date'])

business_df['Month'] = pd.to_datetime(business_df['Month'])
financial_df['Month'] = pd.to_datetime(financial_df['Month'])

# 5. Outlier Detection and Treatment
def detect_outliers(df, columns):
    """Detect outliers using IQR method"""
    outliers_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_info[col] = len(outliers)
    
    return outliers_info

# Check outliers in numerical columns
numerical_cols = ['Age', 'Total_Experience', 'Performance_Rating', 'Avg_Work_Hours', 
                 'Monthly_Salary', 'Market_Salary_Ratio', 'Job_Satisfaction_Score',
                 'Manager_Rating', 'Days_Since_Last_Promotion', 'Commute_Distance_KM']

outliers = detect_outliers(employee_df, numerical_cols)

print("\nOutliers detected:")
for col, count in outliers.items():
    if count > 0:
        print(f"{col}: {count} outliers")

# Cap extreme outliers
def cap_outliers(df, column, lower_percentile=5, upper_percentile=95):
    """Cap outliers at specified percentiles"""
    lower_cap = df[column].quantile(lower_percentile/100)
    upper_cap = df[column].quantile(upper_percentile/100)
    df[column] = df[column].clip(lower=lower_cap, upper=upper_cap)
    return df

# Cap extreme outliers in key columns
for col in ['Avg_Work_Hours', 'Commute_Distance_KM', 'Days_Since_Last_Promotion']:
    employee_df = cap_outliers(employee_df, col)

print("\nOutliers treated by capping at 5th and 95th percentiles")

# 6. Feature Engineering
employee_df['Tenure_Years'] = ((pd.Timestamp.now() - employee_df['Joining_Date']).dt.days / 365).round(1)
employee_df['Experience_Tenure_Ratio'] = (employee_df['Total_Experience'] / (employee_df['Tenure_Years'] + 0.1)).round(2)
employee_df['Salary_Satisfaction'] = (employee_df['Market_Salary_Ratio'] * employee_df['Job_Satisfaction_Score']).round(2)
employee_df['Work_Life_Balance'] = np.where(employee_df['Avg_Work_Hours'] > 50, 'Poor', 
                                   np.where(employee_df['Avg_Work_Hours'] > 45, 'Average', 'Good'))

print("\nNew features created:")
print("- Tenure_Years")
print("- Experience_Tenure_Ratio") 
print("- Salary_Satisfaction")
print("- Work_Life_Balance")

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Set up plotting style
plt.style.use('default')
fig_size = (12, 8)

# 1. Attrition Overview
plt.figure(figsize=(15, 10))

# Attrition Distribution
plt.subplot(2, 3, 1)
attrition_counts = employee_df['Status'].value_counts()
plt.pie(attrition_counts.values, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Employee Status Distribution')

# Department wise attrition
plt.subplot(2, 3, 2)
dept_attrition = employee_df.groupby(['Department', 'Status']).size().unstack()
dept_attrition_rate = (dept_attrition['Resigned'] / (dept_attrition['Active'] + dept_attrition['Resigned']) * 100).sort_values(ascending=False)
dept_attrition_rate.plot(kind='bar', color='coral')
plt.title('Attrition Rate by Department')
plt.ylabel('Attrition Rate (%)')
plt.xticks(rotation=45)

# Age distribution by status
plt.subplot(2, 3, 3)
employee_df.boxplot(column='Age', by='Status', ax=plt.gca())
plt.title('Age Distribution by Status')
plt.suptitle('')

# Job Satisfaction vs Status
plt.subplot(2, 3, 4)
employee_df.boxplot(column='Job_Satisfaction_Score', by='Status', ax=plt.gca())
plt.title('Job Satisfaction by Status')
plt.suptitle('')

# Salary Ratio vs Status
plt.subplot(2, 3, 5)
employee_df.boxplot(column='Market_Salary_Ratio', by='Status', ax=plt.gca())
plt.title('Market Salary Ratio by Status')
plt.suptitle('')

# Lead Time Distribution
plt.subplot(2, 3, 6)
resigned_df = employee_df[employee_df['Status'] == 'Resigned']
plt.hist(resigned_df['Lead_Time'], bins=20, color='lightcoral', alpha=0.7)
plt.title('Lead Time Distribution (Resigned Employees)')
plt.xlabel('Lead Time (Days)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 2. Correlation Analysis
plt.figure(figsize=(12, 8))
numeric_cols = employee_df.select_dtypes(include=[np.number]).columns
correlation_matrix = employee_df[numeric_cols].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix - Employee Features')
plt.tight_layout()
plt.show()

# 3. Key Insights Summary
print("\nKEY INSIGHTS:")

resigned_df = employee_df[employee_df['Status'] == 'Resigned']
active_df = employee_df[employee_df['Status'] == 'Active']

print(f"1. Overall attrition rate: {len(resigned_df)/len(employee_df)*100:.1f}%")

print(f"\n2. Average metrics comparison:")
print(f"   Job Satisfaction - Active: {active_df['Job_Satisfaction_Score'].mean():.1f}, Resigned: {resigned_df['Job_Satisfaction_Score'].mean():.1f}")
print(f"   Market Salary Ratio - Active: {active_df['Market_Salary_Ratio'].mean():.2f}, Resigned: {resigned_df['Market_Salary_Ratio'].mean():.2f}")
print(f"   Manager Rating - Active: {active_df['Manager_Rating'].mean():.1f}, Resigned: {resigned_df['Manager_Rating'].mean():.1f}")

print(f"\n3. Lead time statistics:")
print(f"   Average lead time: {resigned_df['Lead_Time'].mean():.0f} days")
print(f"   Median lead time: {resigned_df['Lead_Time'].median():.0f} days")
print(f"   Lead time range: {resigned_df['Lead_Time'].min():.0f} - {resigned_df['Lead_Time'].max():.0f} days")

# 4. Business & Financial Trends
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(business_df['Month'], business_df['Revenue'], marker='o', linewidth=2)
plt.title('Revenue Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.plot(financial_df['Month'], financial_df['Cash_Flow'], marker='s', linewidth=2, color='green')
plt.title('Cash Flow Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Cash Flow')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Save processed data
employee_df.to_csv('employee_data_processed.csv', index=False)
print(f"\nProcessed data saved to 'employee_data_processed.csv'")
print(f"Shape: {employee_df.shape}")
print("Ready for modeling!")