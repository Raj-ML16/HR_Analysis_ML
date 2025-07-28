import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load datasets - UPDATED to use FIXED data
employee_df = pd.read_csv('employee_data_realistic.csv')  # CHANGED
business_df = pd.read_csv('business_data.csv')
financial_df = pd.read_csv('financial_data.csv')

print("="*60)
print("FIXED DATA PREPROCESSING")
print("="*60)

# 1. Basic Info
print(f"Employee Data Shape: {employee_df.shape}")
print(f"Business Data Shape: {business_df.shape}")
print(f"Financial Data Shape: {financial_df.shape}")

# Check for enhanced features
enhanced_features = ['Intent_To_Stay_12Months', 'Engagement_Survey_Score', 'Meeting_Participation_Score', 
                    'Time_Since_Last_Promotion_Months', 'Training_Completion_Rate']

available_enhanced = [f for f in enhanced_features if f in employee_df.columns]
print(f"Enhanced features detected: {len(available_enhanced)}")

# FIXED: Check for new realistic lead time columns
new_fixed_features = ['Last_Working_Day', 'Notice_Period_Type']
available_fixed = [f for f in new_fixed_features if f in employee_df.columns]
print(f"FIXED features detected: {len(available_fixed)} - {available_fixed}")

# 2. Check Missing Values
print("\nMissing Values in Employee Data:")
missing_emp = employee_df.isnull().sum()
missing_features = missing_emp[missing_emp > 0]
if len(missing_features) > 0:
    print(missing_features)
else:
    print("No missing values found!")

print("\nMissing Values in Business Data:")
missing_bus = business_df.isnull().sum()
missing_bus_features = missing_bus[missing_bus > 0]
if len(missing_bus_features) > 0:
    print(missing_bus_features)
else:
    print("No missing values found!")

print("\nMissing Values in Financial Data:")
missing_fin = financial_df.isnull().sum()
missing_fin_features = missing_fin[missing_fin > 0]
if len(missing_fin_features) > 0:
    print(missing_fin_features)
else:
    print("No missing values found!")

# 3. Handle Missing Values
employee_df['Manager_ID'].fillna('None', inplace=True)
employee_df['Resignation_Date'].fillna('Active', inplace=True)
employee_df['Lead_Time'].fillna(0, inplace=True)

# FIXED: Handle new columns
if 'Last_Working_Day' in employee_df.columns:
    employee_df['Last_Working_Day'].fillna('Active', inplace=True)
if 'Notice_Period_Type' in employee_df.columns:
    employee_df['Notice_Period_Type'].fillna('Unknown', inplace=True)

print("\nAfter handling missing values:")
print(f"Employee missing values: {employee_df.isnull().sum().sum()}")

# 4. Data Types Conversion
employee_df['Joining_Date'] = pd.to_datetime(employee_df['Joining_Date'])
employee_df['Resignation_Date'] = employee_df['Resignation_Date'].replace('Active', pd.NaT)
employee_df['Resignation_Date'] = pd.to_datetime(employee_df['Resignation_Date'])

# FIXED: Handle new Last_Working_Day column
if 'Last_Working_Day' in employee_df.columns:
    employee_df['Last_Working_Day'] = employee_df['Last_Working_Day'].replace('Active', pd.NaT)
    employee_df['Last_Working_Day'] = pd.to_datetime(employee_df['Last_Working_Day'])
    print("✅ Last_Working_Day column processed")

business_df['Month'] = pd.to_datetime(business_df['Month'])
financial_df['Month'] = pd.to_datetime(financial_df['Month'])

# 5. Outlier Detection and Treatment
def detect_outliers(df, columns):
    """Detect outliers using IQR method"""
    outliers_info = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
    
    return outliers_info

# Check outliers in numerical columns (original + enhanced)
numerical_cols = ['Age', 'Total_Experience', 'Performance_Rating', 'Avg_Work_Hours', 
                 'Monthly_Salary', 'Market_Salary_Ratio', 'Job_Satisfaction_Score',
                 'Manager_Rating', 'Days_Since_Last_Promotion', 'Commute_Distance_KM']

# Add enhanced features (excluding integer month field)
enhanced_numerical = [f for f in available_enhanced if f != 'Time_Since_Last_Promotion_Months']
numerical_cols.extend(enhanced_numerical)

outliers = detect_outliers(employee_df, numerical_cols)

print("\nOutliers detected:")
for col, count in outliers.items():
    if count > 0:
        print(f"{col}: {count} outliers")

# Cap extreme outliers
def cap_outliers(df, column, lower_percentile=5, upper_percentile=95):
    """Cap outliers at specified percentiles"""
    if column in df.columns:
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
print("FIXED EXPLORATORY DATA ANALYSIS")
print("="*60)

# Set up plotting style
plt.style.use('default')

# 1. FIXED Attrition Overview
plt.figure(figsize=(16, 12))

# Attrition Distribution
plt.subplot(3, 3, 1)
attrition_counts = employee_df['Status'].value_counts()
plt.pie(attrition_counts.values, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Employee Status Distribution')

# Department wise attrition
plt.subplot(3, 3, 2)
dept_attrition = employee_df.groupby(['Department', 'Status']).size().unstack()
dept_attrition_rate = (dept_attrition['Resigned'] / (dept_attrition['Active'] + dept_attrition['Resigned']) * 100).sort_values(ascending=False)
dept_attrition_rate.plot(kind='bar', color='coral')
plt.title('Attrition Rate by Department')
plt.ylabel('Attrition Rate (%)')
plt.xticks(rotation=45)

# Age distribution by status
plt.subplot(3, 3, 3)
employee_df.boxplot(column='Age', by='Status', ax=plt.gca())
plt.title('Age Distribution by Status')
plt.suptitle('')

# Job Satisfaction vs Status
plt.subplot(3, 3, 4)
employee_df.boxplot(column='Job_Satisfaction_Score', by='Status', ax=plt.gca())
plt.title('Job Satisfaction by Status')
plt.suptitle('')

# Enhanced Feature: Intent to Stay vs Status
plt.subplot(3, 3, 5)
if 'Intent_To_Stay_12Months' in employee_df.columns:
    employee_df.boxplot(column='Intent_To_Stay_12Months', by='Status', ax=plt.gca())
    plt.title('Intent to Stay by Status')
    plt.suptitle('')

# FIXED: Lead Time Distribution
plt.subplot(3, 3, 6)
resigned_df = employee_df[employee_df['Status'] == 'Resigned']
if len(resigned_df) > 0 and 'Lead_Time' in resigned_df.columns:
    plt.hist(resigned_df['Lead_Time'], bins=20, color='lightcoral', alpha=0.7)
    plt.title('FIXED: Notice Period Distribution')
    plt.xlabel('Notice Period (Days)')
    plt.ylabel('Frequency')

# FIXED: Notice Period by Type
plt.subplot(3, 3, 7)
if 'Notice_Period_Type' in resigned_df.columns and len(resigned_df) > 0:
    notice_counts = resigned_df['Notice_Period_Type'].value_counts()
    plt.pie(notice_counts.values, labels=notice_counts.index, autopct='%1.0f')
    plt.title('Notice Period Types')

# FIXED: Lead Time by Department
plt.subplot(3, 3, 8)
if len(resigned_df) > 0 and 'Lead_Time' in resigned_df.columns:
    resigned_df.boxplot(column='Lead_Time', by='Department', ax=plt.gca())
    plt.title('Notice Period by Department')
    plt.suptitle('')
    plt.xticks(rotation=45)

# Market Salary Ratio vs Status
plt.subplot(3, 3, 9)
employee_df.boxplot(column='Market_Salary_Ratio', by='Status', ax=plt.gca())
plt.title('Market Salary Ratio by Status')
plt.suptitle('')

plt.tight_layout()
plt.show()

# 2. Enhanced Correlation Analysis
plt.figure(figsize=(12, 8))

# Focus on key features including enhanced ones
key_features = ['Lead_Time', 'Job_Satisfaction_Score', 'Market_Salary_Ratio', 'Manager_Rating']
key_features.extend([f for f in available_enhanced if f in employee_df.columns])

# Filter to only numeric columns
numeric_key_features = [f for f in key_features if employee_df[f].dtype in ['float64', 'int64']]

correlation_matrix = employee_df[numeric_key_features].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('FIXED Features Correlation Matrix')
plt.tight_layout()
plt.show()

# 3. FIXED Key Insights Summary
print("\nFIXED KEY INSIGHTS:")

resigned_df = employee_df[employee_df['Status'] == 'Resigned']
active_df = employee_df[employee_df['Status'] == 'Active']

print(f"1. Overall attrition rate: {len(resigned_df)/len(employee_df)*100:.1f}%")

print(f"\n2. Average metrics comparison:")
print(f"   Job Satisfaction - Active: {active_df['Job_Satisfaction_Score'].mean():.1f}, Resigned: {resigned_df['Job_Satisfaction_Score'].mean():.1f}")
print(f"   Market Salary Ratio - Active: {active_df['Market_Salary_Ratio'].mean():.2f}, Resigned: {resigned_df['Market_Salary_Ratio'].mean():.2f}")
print(f"   Manager Rating - Active: {active_df['Manager_Rating'].mean():.1f}, Resigned: {resigned_df['Manager_Rating'].mean():.1f}")

# Enhanced features comparison
for feature in available_enhanced:
    active_avg = active_df[feature].mean()
    resigned_avg = resigned_df[feature].mean()
    print(f"   {feature} - Active: {active_avg:.2f}, Resigned: {resigned_avg:.2f}")

# FIXED: Lead time statistics
if len(resigned_df) > 0 and 'Lead_Time' in resigned_df.columns:
    print(f"\n3. FIXED Lead time statistics:")
    print(f"   Average notice period: {resigned_df['Lead_Time'].mean():.0f} days")
    print(f"   Median notice period: {resigned_df['Lead_Time'].median():.0f} days")
    print(f"   Notice period range: {resigned_df['Lead_Time'].min():.0f} - {resigned_df['Lead_Time'].max():.0f} days")
    
    # Notice period by type
    if 'Notice_Period_Type' in resigned_df.columns:
        print(f"\n4. FIXED Notice period by type:")
        notice_stats = resigned_df.groupby('Notice_Period_Type')['Lead_Time'].agg(['count', 'mean', 'std']).round(1)
        for notice_type, stats in notice_stats.iterrows():
            print(f"   {notice_type}: {stats['count']} employees, avg {stats['mean']} days, std {stats['std']} days")

# Enhanced timeline correlation analysis
print(f"\n5. FIXED TIMELINE CORRELATION:")
resigned_with_leadtime = resigned_df[resigned_df['Lead_Time'] > 0]

if len(resigned_with_leadtime) > 10:
    for feature in available_enhanced:
        if feature in resigned_with_leadtime.columns:
            correlation = resigned_with_leadtime[feature].corr(resigned_with_leadtime['Lead_Time'])
            print(f"   {feature} vs Lead_Time: {correlation:.3f}")
    
    # Compare with original
    original_corr = resigned_with_leadtime['Job_Satisfaction_Score'].corr(resigned_with_leadtime['Lead_Time'])
    print(f"   Job_Satisfaction_Score vs Lead_Time: {original_corr:.3f} (original)")

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
print(f"\nFIXED processed data saved to 'employee_data_processed.csv'")
print(f"Shape: {employee_df.shape}")
print(f"🎯 Ready for FIXED attrition modeling!")