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

def get_standard_notice_period(designation, department):
    """Get standard notice period based on role and department (REALISTIC APPROACH)"""
    
    # Executive level (C-suite, VPs, Directors)
    if any(title in designation for title in ['Manager', 'Director', 'VP', 'CFO']):
        base_notice = 90  # 3 months
        notice_type = 'Executive'
        
    # Senior technical/professional roles
    elif any(title in designation for title in ['Tech Lead', 'Senior', 'Lead']):
        base_notice = 60  # 2 months
        notice_type = 'Senior'
        
    # Professional roles in high-skill departments
    elif department in ['Engineering', 'Finance']:
        base_notice = 30  # 1 month
        notice_type = 'Professional'
        
    # Standard roles
    else:
        base_notice = 14  # 2 weeks
        notice_type = 'Standard'
    
    # Add realistic variation (±20% but at least 7 days)
    variation_range = int(base_notice * 0.2)
    actual_notice = base_notice + random.randint(-variation_range, variation_range)
    
    return max(7, actual_notice), notice_type

def apply_notice_variations(standard_notice, employee_data):
    """Apply realistic variations to standard notice periods based on employee circumstances"""
    
    actual_notice = standard_notice
    
    # High performers often give longer notice for better handover
    if employee_data['Performance_Rating'] > 4.0:
        if random.random() < 0.3:  # 30% chance
            actual_notice += random.randint(7, 21)  # Extra 1-3 weeks
    
    # Emergency personal situations (family, health, urgent opportunities)
    if random.random() < 0.08:  # 8% chance - realistic emergency rate
        actual_notice = max(7, actual_notice - random.randint(7, 21))
        # Note: 7 days minimum (legal/contractual requirement)
    
    # Significantly underpaid employees might get early release
    if employee_data['Market_Salary_Ratio'] < 0.80:  # Very underpaid
        if random.random() < 0.25:  # 25% chance
            actual_notice = max(14, actual_notice - random.randint(7, 14))
    
    # Long tenure employees often give longer notice (loyalty factor)
    tenure_years = (datetime.now().date() - employee_data['Joining_Date']).days / 365.25
    if tenure_years > 5 and random.random() < 0.4:  # 40% chance for 5+ year employees
        actual_notice += random.randint(7, 14)
    
    # Very low engagement/satisfaction might lead to shorter notice
    if employee_data['Job_Satisfaction_Score'] < 3.0 and employee_data['Intent_To_Stay_12Months'] < 3.0:
        if random.random() < 0.2:  # 20% chance for very disengaged employees
            actual_notice = max(14, actual_notice - random.randint(0, 14))
    
    # Senior employees with competing offers often negotiate notice period
    if 'Manager' in employee_data['Designation'] and employee_data['Market_Salary_Ratio'] < 0.90:
        if random.random() < 0.3:  # 30% chance
            # Could be shorter (competing offer pressure) or longer (negotiated handover)
            adjustment = random.choice([-14, -7, 7, 14])
            actual_notice = max(21, actual_notice + adjustment)  # Managers minimum 3 weeks
    
    return actual_notice

def generate_enhanced_employee_data():
    """Generate employee dataset with realistic patterns + TOP 5 timeline features"""
    
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
        
        # === TOP 5 ENHANCED FEATURES FOR TIMELINE PREDICTION ===
        
        # 1. Intent_To_Stay_12Months (🥇 STRONGEST PREDICTOR)
        intent_to_stay_12months = np.clip(np.random.normal(7.0, 2.0), 1, 10)
        
        # 2. Engagement_Survey_Score (🥈 BEHAVIORAL SIGNAL)
        engagement_survey_score = np.clip(np.random.normal(6.8, 1.8), 1, 10)
        
        # 3. Meeting_Participation_Score (🥉 OBSERVABLE DECLINE)
        meeting_participation_score = np.clip(np.random.normal(7.5, 1.8), 1, 10)
        
        # 4. Time_Since_Last_Promotion_Months (🏅 CAREER STAGNATION)
        time_since_last_promotion_months = np.random.randint(6, 60)
        
        # 5. Training_Completion_Rate (🏅 ROLE COMMITMENT)
        training_completion_rate = np.clip(np.random.normal(0.75, 0.25), 0, 1)
        
        employee = {
            # Basic Info (Original)
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
            
            # Performance & Work (Original)
            'Performance_Rating': round(performance_rating, 1),
            'Avg_Work_Hours': round(avg_work_hours, 1),
            'Project_Workload': project_workload,
            'Remote_Work_Days_Monthly': remote_work_days,
            
            # Compensation (Original)
            'Monthly_Salary': monthly_salary,
            'Market_Salary_Ratio': round(market_salary_ratio, 2),
            'Salary_Change_%': round(salary_change_pct, 1),
            'Bonus_Last_Year': bonus_last_year,
            
            # Satisfaction & Engagement (Original)
            'Job_Satisfaction_Score': round(job_satisfaction, 1),
            'Manager_Rating': round(manager_rating, 1),
            
            # Behavioral Indicators (Original)
            'Internal_Job_Applications': internal_job_applications,
            'Days_Since_Last_Promotion': days_since_promotion,
            'Leaves_Taken_Last_Year': leaves_taken,
            'Commute_Distance_KM': round(commute_distance, 1),
            'Team_Size': team_size,
            
            # === NEW TOP 5 ENHANCED FEATURES ===
            'Intent_To_Stay_12Months': round(intent_to_stay_12months, 1),
            'Engagement_Survey_Score': round(engagement_survey_score, 1),
            'Meeting_Participation_Score': round(meeting_participation_score, 1),
            'Time_Since_Last_Promotion_Months': time_since_last_promotion_months,
            'Training_Completion_Rate': round(training_completion_rate, 2)
        }
        
        employees.append(employee)
    
    # Convert to DataFrame
    df = pd.DataFrame(employees)
    
    # Create ENHANCED attrition patterns with REALISTIC lead time calculation
    df = create_enhanced_attrition_patterns_FIXED(df)
    
    return df

def create_enhanced_attrition_patterns_FIXED(df):
    """Create enhanced attrition patterns with REALISTIC lead time calculation"""
    
    # Calculate attrition probability with ENHANCED weights
    attrition_prob = np.random.random(len(df)) * 0.1  # Base random component
    
    # Original influences (SUBTLE)
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
    
    # ENHANCED influences (STRONGER for timeline prediction)
    attrition_prob += np.where(df['Intent_To_Stay_12Months'] < 4, 0.35, 0)  # Strongest
    attrition_prob += np.where(df['Intent_To_Stay_12Months'] < 6, 0.20, 0)
    attrition_prob += np.where(df['Engagement_Survey_Score'] < 4, 0.30, 0)
    attrition_prob += np.where(df['Engagement_Survey_Score'] < 6, 0.15, 0)
    attrition_prob += np.where(df['Meeting_Participation_Score'] < 5, 0.20, 0)
    attrition_prob += np.where(df['Meeting_Participation_Score'] < 7, 0.10, 0)
    attrition_prob += np.where(df['Time_Since_Last_Promotion_Months'] > 36, 0.15, 0)
    attrition_prob += np.where(df['Training_Completion_Rate'] < 0.5, 0.12, 0)
    
    # Ensure probabilities are between 0 and 1
    attrition_prob = np.clip(attrition_prob, 0, 0.9)
    
    # Select employees for attrition based on probability
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
    
    # === FIXED APPROACH: REALISTIC LEAD TIME CALCULATION ===
    df['Resignation_Date'] = None
    df['Last_Working_Day'] = None
    df['Lead_Time'] = None
    df['Notice_Period_Type'] = None
    
    departed_mask = df['Status'] == 'Resigned'
    
    print(f"🔧 Generating REALISTIC resignation timelines for {departed_mask.sum()} employees...")
    
    for idx in df[departed_mask].index:
        employee_data = df.loc[idx]
        
        # Step 1: Generate resignation date (within last 2 years)
        resignation_date = fake.date_between(start_date='-2y', end_date='-1m')
        
        # Step 2: Get standard notice period based on role
        standard_notice_days, notice_type = get_standard_notice_period(
            employee_data['Designation'], 
            employee_data['Department']
        )
        
        # Step 3: Apply realistic variations based on employee circumstances
        actual_notice_days = apply_notice_variations(standard_notice_days, employee_data)
        
        # Step 4: Calculate last working day
        last_working_day = resignation_date + timedelta(days=actual_notice_days)
        
        # Step 5: Store the REALISTIC data
        df.loc[idx, 'Resignation_Date'] = resignation_date
        df.loc[idx, 'Last_Working_Day'] = last_working_day
        df.loc[idx, 'Lead_Time'] = actual_notice_days  # REAL LEAD TIME = LWD - Resignation Date
        df.loc[idx, 'Notice_Period_Type'] = notice_type
    
    # Apply ENHANCED behavioral decline for departed employees
    print("🔧 Applying behavioral decline patterns...")
    for idx in df[departed_mask].index:
        lead_time = df.loc[idx, 'Lead_Time']
        
        # Timeline-based decline (shorter lead time = more decline)
        if lead_time < 30:  # Short notice - significant decline
            decline_factor = np.random.uniform(0.6, 0.8)
            enhanced_decline = np.random.uniform(0.5, 0.7)
        elif lead_time < 60:  # Medium notice - moderate decline
            decline_factor = np.random.uniform(0.75, 0.9)
            enhanced_decline = np.random.uniform(0.7, 0.85)
        else:  # Long notice - subtle decline
            decline_factor = np.random.uniform(0.85, 0.95)
            enhanced_decline = np.random.uniform(0.8, 0.95)
        
        # Apply decline to original features
        if np.random.random() < 0.7:  # 70% chance to have satisfaction decline
            df.loc[idx, 'Job_Satisfaction_Score'] *= decline_factor
            df.loc[idx, 'Job_Satisfaction_Score'] = max(1, df.loc[idx, 'Job_Satisfaction_Score'])
        
        if np.random.random() < 0.5:  # 50% chance to have manager rating decline
            df.loc[idx, 'Manager_Rating'] *= decline_factor
            df.loc[idx, 'Manager_Rating'] = max(1, df.loc[idx, 'Manager_Rating'])
        
        if np.random.random() < 0.4:  # 40% chance to have more internal applications
            df.loc[idx, 'Internal_Job_Applications'] = min(3, df.loc[idx, 'Internal_Job_Applications'] + np.random.poisson(0.5))
        
        # Apply ENHANCED decline to TOP 5 features (stronger correlations)
        if np.random.random() < 0.9:  # 90% chance for intent decline
            df.loc[idx, 'Intent_To_Stay_12Months'] *= enhanced_decline
            df.loc[idx, 'Intent_To_Stay_12Months'] = max(1, df.loc[idx, 'Intent_To_Stay_12Months'])
        
        if np.random.random() < 0.8:  # 80% chance for engagement decline
            df.loc[idx, 'Engagement_Survey_Score'] *= enhanced_decline
            df.loc[idx, 'Engagement_Survey_Score'] = max(1, df.loc[idx, 'Engagement_Survey_Score'])
        
        if np.random.random() < 0.7:  # 70% chance for meeting participation decline
            df.loc[idx, 'Meeting_Participation_Score'] *= enhanced_decline
            df.loc[idx, 'Meeting_Participation_Score'] = max(1, df.loc[idx, 'Meeting_Participation_Score'])
        
        if np.random.random() < 0.6:  # 60% chance for training completion decline
            df.loc[idx, 'Training_Completion_Rate'] *= enhanced_decline
            df.loc[idx, 'Training_Completion_Rate'] = max(0.1, df.loc[idx, 'Training_Completion_Rate'])
    
    return df

def generate_business_data():
    """Generate business performance data (same as original)"""
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
    """Generate financial indicators data (same as original)"""
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
print("🚀 Generating FIXED Employee Dataset with REALISTIC Lead Times...")
print("="*70)
employee_df = generate_enhanced_employee_data()

business_df = generate_business_data()
financial_df = generate_financial_data()

# ENHANCED SUMMARY WITH LEAD TIME ANALYSIS
print("\n" + "="*70)
print("📊 FIXED DATASET SUMMARY")
print("="*70)

print(f"✅ Dataset created: {len(employee_df)} employees, {len(employee_df.columns)} features")
print(f"✅ Active: {len(employee_df[employee_df['Status'] == 'Active'])}, Resigned: {len(employee_df[employee_df['Status'] == 'Resigned'])}")

# Lead time analysis
resigned_df = employee_df[employee_df['Status'] == 'Resigned']
if len(resigned_df) > 0:
    print(f"\n📅 REALISTIC LEAD TIME ANALYSIS:")
    print(f"   Average notice period: {resigned_df['Lead_Time'].mean():.1f} days")
    print(f"   Median notice period: {resigned_df['Lead_Time'].median():.1f} days")
    print(f"   Notice period range: {resigned_df['Lead_Time'].min():.0f} - {resigned_df['Lead_Time'].max():.0f} days")
    
    # Notice type distribution
    notice_distribution = resigned_df['Notice_Period_Type'].value_counts()
    print(f"\n   Notice Period Types:")
    for notice_type, count in notice_distribution.items():
        avg_days = resigned_df[resigned_df['Notice_Period_Type'] == notice_type]['Lead_Time'].mean()
        print(f"     {notice_type}: {count} employees (avg: {avg_days:.1f} days)")

# Save datasets
employee_df.to_csv('employee_data_realistic.csv', index=False)
business_df.to_csv('business_data.csv', index=False)
financial_df.to_csv('financial_data.csv', index=False)

print(f"\n✅ FILES SAVED:")
print(f"   📄 employee_data_realistic.csv")
print(f"   📄 business_data.csv") 
print(f"   📄 financial_data.csv")

print("="*70)
print("🚀 READY FOR FIXED ATTRITION MODELING!")
print("="*70)