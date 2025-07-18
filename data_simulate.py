import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()

# Configuration
TOTAL_EMPLOYEES = 3000
DEPARTED_EMPLOYEES = 900
ACTIVE_EMPLOYEES = 2100

print("Generating IMPROVED synthetic dataset with realistic patterns...")

# Department and designation mapping
DEPARTMENTS = {
    'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
    'Sales': ['Sales Executive', 'Senior Sales', 'Sales Manager', 'Regional Manager'],
    'Marketing': ['Marketing Executive', 'Marketing Manager', 'Content Manager', 'Brand Manager'],
    'HR': ['HR Executive', 'HR Manager', 'Recruiter', 'HR Director'],
    'Finance': ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'CFO'],
    'Operations': ['Operations Executive', 'Operations Manager', 'Process Manager', 'VP Operations']
}

# Salary ranges by department and level
SALARY_RANGES = {
    'Engineering': {'Software': (50000, 80000), 'Senior': (80000, 120000), 'Tech': (120000, 180000), 'Engineering': (150000, 220000)},
    'Sales': {'Sales': (40000, 70000), 'Senior': (70000, 100000), 'Regional': (140000, 200000)},
    'Marketing': {'Marketing': (45000, 75000), 'Content': (60000, 90000), 'Brand': (90000, 140000)},
    'HR': {'HR': (40000, 65000), 'Recruiter': (45000, 70000), 'Director': (120000, 180000)},
    'Finance': {'Financial': (50000, 80000), 'Senior': (80000, 120000), 'Finance': (120000, 170000), 'CFO': (200000, 300000)},
    'Operations': {'Operations': (45000, 70000), 'Process': (80000, 120000), 'VP': (150000, 250000)}
}

def generate_realistic_employee_data():
    """Generate employee dataset with realistic, overlapping patterns"""
    
    employees = []
    
    # Generate base employee data
    for i in range(1, TOTAL_EMPLOYEES + 1):
        # Basic demographics
        gender = np.random.choice(['Male', 'Female'], p=[0.6, 0.4])
        age = np.random.randint(22, 60)
        location = np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune'], 
                                   p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        education = np.random.choice(['Bachelor', 'Master', 'PhD'], p=[0.6, 0.35, 0.05])
        
        # Department and designation
        department = np.random.choice(list(DEPARTMENTS.keys()))
        designation = np.random.choice(DEPARTMENTS[department])
        
        # Experience and tenure
        total_experience = max(0, age - 22 + np.random.randint(-2, 3))
        joining_date = fake.date_between(start_date='-8y', end_date='-6m')
        
        # Performance and ratings (normal distribution)
        performance_rating = np.clip(np.random.normal(3.5, 0.7), 1, 5)
        
        # Work characteristics
        avg_work_hours = np.clip(np.random.normal(45, 6), 35, 65)
        
        project_workload = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.6, 0.2])
        remote_work_days = np.random.randint(0, 5)
        
        # Salary calculation
        salary_key = designation.split()[0] if len(designation.split()) > 1 else designation
        if salary_key not in SALARY_RANGES[department]:
            salary_key = list(SALARY_RANGES[department].keys())[0]
        
        salary_range = SALARY_RANGES[department][salary_key]
        monthly_salary = np.random.randint(salary_range[0], salary_range[1])
        
        # Market salary ratio (REALISTIC - most people are close to market rate)
        market_salary_ratio = np.clip(np.random.normal(0.98, 0.12), 0.75, 1.25)
        
        # Engagement and satisfaction (REALISTIC - most people are moderately satisfied)
        job_satisfaction = np.clip(np.random.normal(6.5, 1.8), 1, 10)
        manager_rating = np.clip(np.random.normal(7.0, 1.5), 1, 10)
        
        # Behavioral indicators (REALISTIC - low base rates)
        internal_job_applications = np.random.poisson(0.3)  # Lower base rate
        days_since_promotion = np.random.randint(90, 1200)  # More realistic range
        leaves_taken = np.random.randint(8, 22)
        
        # Commute and team factors
        commute_distance = np.clip(np.random.gamma(2, 4), 1, 40)
        team_size = np.random.randint(5, 20)
        
        # Financial factors
        salary_change_pct = np.clip(np.random.normal(6, 4), -5, 20)
        bonus_last_year = np.random.choice([0, 1], p=[0.4, 0.6])
        
        # Manager assignment (fixed)
        if i <= 10:
            manager_id = None
        else:
            manager_id = np.random.randint(1, min(11, i))
        
        employee = {
            'Employee_ID': f'EMP_{i:04d}',
            'Name': fake.name(),
            'Gender': gender,
            'Age': age,
            'Location': location,
            'Education': education,
            'Total_Experience': total_experience,
            'Department': department,
            'Designation': designation,
            'Manager_ID': f'EMP_{manager_id:04d}' if manager_id else None,
            'Joining_Date': joining_date,
            'Performance_Rating': round(performance_rating, 1),
            'Avg_Work_Hours': round(avg_work_hours, 1),
            'Project_Workload': project_workload,
            'Remote_Work_Days_Monthly': remote_work_days,
            'Monthly_Salary': monthly_salary,
            'Market_Salary_Ratio': round(market_salary_ratio, 2),
            'Salary_Change_%': round(salary_change_pct, 1),
            'Bonus_Last_Year': bonus_last_year,
            'Job_Satisfaction_Score': round(job_satisfaction, 1),
            'Manager_Rating': round(manager_rating, 1),
            'Internal_Job_Applications': internal_job_applications,
            'Days_Since_Last_Promotion': days_since_promotion,
            'Leaves_Taken_Last_Year': leaves_taken,
            'Commute_Distance_KM': round(commute_distance, 1),
            'Team_Size': team_size
        }
        
        employees.append(employee)
    
    # Convert to DataFrame
    df = pd.DataFrame(employees)
    
    # Create REALISTIC attrition patterns (with overlap)
    df = create_realistic_attrition_patterns(df)
    
    return df

def create_realistic_attrition_patterns(df):
    """Create realistic attrition patterns with natural overlap"""
    
    # Calculate attrition probability with REALISTIC weights
    # Most factors should have SUBTLE influence, not perfect separation
    
    attrition_prob = np.random.random(len(df)) * 0.1  # Base random component
    
    # Add SUBTLE influences (not perfect predictors)
    attrition_prob += np.where(df['Job_Satisfaction_Score'] < 4, 0.25, 0)
    attrition_prob += np.where(df['Job_Satisfaction_Score'] < 6, 0.15, 0)
    attrition_prob += np.where(df['Market_Salary_Ratio'] < 0.85, 0.20, 0)
    attrition_prob += np.where(df['Market_Salary_Ratio'] < 0.95, 0.10, 0)
    attrition_prob += np.where(df['Manager_Rating'] < 5, 0.18, 0)
    attrition_prob += np.where(df['Manager_Rating'] < 7, 0.08, 0)
    attrition_prob += np.where(df['Days_Since_Last_Promotion'] > 900, 0.12, 0)
    attrition_prob += np.where(df['Internal_Job_Applications'] > 1, 0.15, 0)
    attrition_prob += np.where(df['Performance_Rating'] < 2.5, 0.10, 0)
    attrition_prob += np.where(df['Avg_Work_Hours'] > 55, 0.08, 0)
    attrition_prob += np.where(df['Commute_Distance_KM'] > 25, 0.05, 0)
    
    # Ensure probabilities are between 0 and 1
    attrition_prob = np.clip(attrition_prob, 0, 0.9)
    
    # Select employees for attrition based on probability
    # Create a more realistic selection
    sorted_indices = np.argsort(attrition_prob)[::-1]  # Highest probability first
    
    # Take top employees but add some randomness
    attrition_indices = []
    for i in range(DEPARTED_EMPLOYEES):
        if i < DEPARTED_EMPLOYEES * 0.8:  # 80% from highest probability
            attrition_indices.append(sorted_indices[i])
        else:  # 20% random selection for realism
            remaining_indices = sorted_indices[DEPARTED_EMPLOYEES:]
            random_idx = np.random.choice(remaining_indices)
            attrition_indices.append(random_idx)
    
    df['Status'] = 'Active'
    df.loc[attrition_indices, 'Status'] = 'Resigned'
    
    # Calculate resignation date and lead time for departed employees
    df['Resignation_Date'] = None
    df['Lead_Time'] = None
    
    departed_mask = df['Status'] == 'Resigned'
    
    # Generate resignation dates (within last 2 years)
    for idx in df[departed_mask].index:
        resignation_date = fake.date_between(start_date='-2y', end_date='-1m')
        df.loc[idx, 'Resignation_Date'] = resignation_date
        
        # Calculate lead time with MORE REALISTIC patterns
        base_lead_time = 90  # Base 3 months
        
        # SUBTLE adjustments (not perfect predictors)
        satisfaction_factor = max(0, (6 - df.loc[idx, 'Job_Satisfaction_Score']) * 8)
        salary_factor = max(0, (0.95 - df.loc[idx, 'Market_Salary_Ratio']) * 80)
        manager_factor = max(0, (7 - df.loc[idx, 'Manager_Rating']) * 5)
        
        # Add randomness for realism
        random_factor = np.random.normal(0, 20)
        
        lead_time = base_lead_time + satisfaction_factor + salary_factor + manager_factor + random_factor
        lead_time = max(30, min(300, lead_time))  # Realistic range
        
        df.loc[idx, 'Lead_Time'] = int(lead_time)
    
    # Apply MINIMAL behavioral decline for departed employees (realistic)
    for idx in df[departed_mask].index:
        # Only SLIGHT decline, not dramatic changes
        decline_factor = np.random.uniform(0.85, 0.95)  # Much smaller decline
        
        # Only decline some metrics, not all
        if np.random.random() < 0.7:  # 70% chance to have satisfaction decline
            df.loc[idx, 'Job_Satisfaction_Score'] *= decline_factor
            df.loc[idx, 'Job_Satisfaction_Score'] = max(1, df.loc[idx, 'Job_Satisfaction_Score'])
        
        if np.random.random() < 0.5:  # 50% chance to have manager rating decline
            df.loc[idx, 'Manager_Rating'] *= decline_factor
            df.loc[idx, 'Manager_Rating'] = max(1, df.loc[idx, 'Manager_Rating'])
        
        if np.random.random() < 0.4:  # 40% chance to have more internal applications
            df.loc[idx, 'Internal_Job_Applications'] = min(3, df.loc[idx, 'Internal_Job_Applications'] + np.random.poisson(0.5))
    
    return df

def generate_business_data():
    """Generate business performance data (same as before)"""
    business_data = []
    start_date = datetime.now() - timedelta(days=36*30)
    
    for i in range(36):
        month_date = start_date + timedelta(days=i*30)
        base_revenue = 10000000
        growth_trend = 1.02 ** i
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)
        noise = np.random.normal(1, 0.05)
        revenue = base_revenue * growth_trend * seasonal_factor * noise
        
        business_record = {
            'Month': month_date.strftime('%Y-%m'),
            'Revenue': round(revenue, 2),
            'Profit_Margin': round(np.clip(np.random.normal(0.15, 0.03), 0.05, 0.25), 3),
            'Customer_Acquisition': np.random.poisson(500),
            'Customer_Retention_Rate': round(np.clip(np.random.normal(0.85, 0.05), 0.7, 0.95), 3),
            'Market_Share': round(np.clip(np.random.normal(0.12, 0.02), 0.08, 0.18), 3),
            'Active_Projects': np.random.randint(50, 150),
            'New_Contracts': np.random.randint(10, 40),
            'Employee_Count': np.random.randint(2800, 3200)
        }
        business_data.append(business_record)
    
    return pd.DataFrame(business_data)

def generate_financial_data():
    """Generate financial indicators data (same as before)"""
    financial_data = []
    start_date = datetime.now() - timedelta(days=36*30)
    
    for i in range(36):
        month_date = start_date + timedelta(days=i*30)
        
        financial_record = {
            'Month': month_date.strftime('%Y-%m'),
            'Cash_Flow': round(np.random.normal(2000000, 500000), 2),
            'Debt_to_Equity_Ratio': round(np.clip(np.random.normal(0.4, 0.1), 0.1, 0.8), 3),
            'ROA': round(np.clip(np.random.normal(0.08, 0.02), 0.02, 0.15), 3),
            'ROE': round(np.clip(np.random.normal(0.12, 0.03), 0.05, 0.20), 3),
            'Working_Capital': round(np.random.normal(5000000, 1000000), 2),
            'RD_Investment': round(np.random.normal(1500000, 300000), 2),
            'CAPEX': round(np.random.normal(800000, 200000), 2),
            'Stock_Price': round(max(50, 100 + i * 2 + np.random.normal(0, 5)), 2),
            'Market_Cap': round((100 + i * 2 + np.random.normal(0, 5)) * 1000000, 2),
            'Dividend_Yield': round(np.clip(np.random.normal(0.03, 0.01), 0.01, 0.06), 3)
        }
        financial_data.append(financial_record)
    
    return pd.DataFrame(financial_data)

# Generate all datasets
print("Generating Employee Dataset with realistic patterns...")
employee_df = generate_realistic_employee_data()

print("Generating Business Dataset...")
business_df = generate_business_data()

print("Generating Financial Dataset...")
financial_df = generate_financial_data()

# Save datasets
employee_df.to_csv('employee_data_realistic.csv', index=False)
business_df.to_csv('business_data.csv', index=False)
financial_df.to_csv('financial_data.csv', index=False)

# Display summary statistics
print("\n" + "="*60)
print("REALISTIC DATASET SUMMARY")
print("="*60)

print(f"\nEMPLOYEE DATA:")
print(f"Total Employees: {len(employee_df)}")
print(f"Active Employees: {len(employee_df[employee_df['Status'] == 'Active'])}")
print(f"Resigned Employees: {len(employee_df[employee_df['Status'] == 'Resigned'])}")

# Analyze realistic patterns
active_df = employee_df[employee_df['Status'] == 'Active']
resigned_df = employee_df[employee_df['Status'] == 'Resigned']

print(f"\nREALISTIC PATTERN ANALYSIS:")
print(f"Job Satisfaction - Active: {active_df['Job_Satisfaction_Score'].mean():.1f}, Resigned: {resigned_df['Job_Satisfaction_Score'].mean():.1f}")
print(f"Market Salary Ratio - Active: {active_df['Market_Salary_Ratio'].mean():.2f}, Resigned: {resigned_df['Market_Salary_Ratio'].mean():.2f}")
print(f"Manager Rating - Active: {active_df['Manager_Rating'].mean():.1f}, Resigned: {resigned_df['Manager_Rating'].mean():.1f}")

# Check overlap (this should be significant for realistic data)
low_sat_active = len(active_df[active_df['Job_Satisfaction_Score'] < 5])
low_sat_resigned = len(resigned_df[resigned_df['Job_Satisfaction_Score'] < 5])
print(f"\nOVERLAP ANALYSIS (indicates realism):")
print(f"Active employees with low satisfaction (<5): {low_sat_active}")
print(f"Resigned employees with low satisfaction (<5): {low_sat_resigned}")

# Lead time analysis
lead_times = resigned_df['Lead_Time'].dropna()
print(f"\nLEAD TIME ANALYSIS:")
print(f"Average: {lead_times.mean():.0f} days")
print(f"Range: {lead_times.min():.0f} - {lead_times.max():.0f} days")
print(f"Std: {lead_times.std():.0f} days")

print("\n" + "="*60)
print("REALISTIC FILES SAVED:")
print("- employee_data_realistic.csv")
print("- business_data.csv (updated)")
print("- financial_data.csv (updated)")
print("="*60)

# Calculate risk distribution for dashboard validation
print(f"\nRISK DISTRIBUTION VALIDATION:")
active_employees = employee_df[employee_df['Status'] == 'Active']

# Calculate risk scores using the same formula as dashboard
risk_scores = (
    (10 - active_employees['Job_Satisfaction_Score']) / 10 * 0.3 +
    (1 - active_employees['Market_Salary_Ratio']) * 0.25 +
    (10 - active_employees['Manager_Rating']) / 10 * 0.2 +
    (active_employees['Days_Since_Last_Promotion'] / 1000) * 0.15 +
    (active_employees['Internal_Job_Applications'] / 5) * 0.1
)

print(f"Risk Score Statistics:")
print(f"Min: {risk_scores.min():.3f}")
print(f"Max: {risk_scores.max():.3f}")
print(f"Mean: {risk_scores.mean():.3f}")
print(f"50th percentile: {np.percentile(risk_scores, 50):.3f}")
print(f"70th percentile: {np.percentile(risk_scores, 70):.3f}")
print(f"80th percentile: {np.percentile(risk_scores, 80):.3f}")

# Recommended thresholds
high_threshold = np.percentile(risk_scores, 80)
medium_threshold = np.percentile(risk_scores, 50)

print(f"\nRECOMMENDED DASHBOARD THRESHOLDS:")
print(f"High Risk: > {high_threshold:.2f}")
print(f"Medium Risk: {medium_threshold:.2f} - {high_threshold:.2f}")
print(f"Low Risk: < {medium_threshold:.2f}")

high_risk_count = len(risk_scores[risk_scores > high_threshold])
medium_risk_count = len(risk_scores[(risk_scores > medium_threshold) & (risk_scores <= high_threshold)])
low_risk_count = len(risk_scores[risk_scores <= medium_threshold])

print(f"\nEXPECTED COUNTS WITH NEW THRESHOLDS:")
print(f"High Risk: {high_risk_count}")
print(f"Medium Risk: {medium_risk_count}")
print(f"Low Risk: {low_risk_count}")

print(f"\n🎯 NEXT STEPS:")
print("1. Use 'employee_data_realistic.csv' for modeling")
print("2. Update dashboard risk thresholds to:")
print(f"   - High: > {high_threshold:.2f}")
print(f"   - Medium: {medium_threshold:.2f} - {high_threshold:.2f}")
print("3. Fix dashboard plotting errors")
print("4. Expect more realistic model performance (AUC 0.70-0.85)")