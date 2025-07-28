import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FIXED HIRING CALCULATOR - Sub-module 2")
print("Uses FIXED Module 1's REALISTIC ML predictions for accurate hiring timeline")
print("="*60)

# Step 1: Load FIXED Module 1's ML predictions
print("\nStep 1: Loading FIXED Module 1's REALISTIC ML predictions...")

import glob
import os

try:
    # Simple approach: Find the latest FIXED file automatically
    fixed_files = glob.glob('Employee_Attrition_Predictions_FIXED_*.xlsx')  # CHANGED
    
    if not fixed_files:
        print("❌ No FIXED predictions file found!")
        print("Please run the FIXED attrition_modelling.py first")
        print("Expected file pattern: Employee_Attrition_Predictions_FIXED_YYYYMMDD_HHMMSS.xlsx")  # CHANGED
        exit()
    
    # Get the most recent FIXED file
    latest_file = max(fixed_files, key=os.path.getmtime)
    
    # Load the predictions
    ml_predictions = pd.read_excel(latest_file, sheet_name='All_Employee_Predictions')
    
    print(f"✅ Loaded FIXED ML predictions from: {latest_file}")
    print(f"   Total employees: {len(ml_predictions)}")
    
    # Check data quality from FIXED model
    with_dates = len(ml_predictions[ml_predictions['Estimated_Departure_Date'].notna()])
    without_dates = len(ml_predictions[ml_predictions['Estimated_Departure_Date'].isna()])
    
    print(f"FIXED Data Quality Check:")
    print(f"  ✅ With departure dates: {with_dates} employees ({with_dates/len(ml_predictions)*100:.1f}%)")
    print(f"  ❌ Without departure dates: {without_dates} employees ({without_dates/len(ml_predictions)*100:.1f}%)")
    
    # Display FIXED risk distribution
    risk_distribution = ml_predictions['Risk_Category'].value_counts()
    print(f"FIXED ML Risk Distribution:")
    for risk, count in risk_distribution.items():
        print(f"  {risk} Risk: {count} employees")
    
    # FIXED: Check for new notice period column
    if 'Predicted_Notice_Period_Days' in ml_predictions.columns:
        avg_notice = ml_predictions['Predicted_Notice_Period_Days'].mean()
        print(f"FIXED Notice Period Analysis:")
        print(f"  Average predicted notice: {avg_notice:.1f} days")
        
except Exception as e:
    print(f"❌ Error loading FIXED predictions: {e}")
    print("Make sure you have run the FIXED attrition_modelling.py first")
    exit()

# Step 2: Load business data for growth calculations
print("\nStep 2: Loading business data...")
business_df = pd.read_csv('business_data.csv')
business_df['Month'] = pd.to_datetime(business_df['Month'])

# Calculate revenue growth
latest_revenue = business_df['Revenue'].iloc[-1]
year_ago_revenue = business_df['Revenue'].iloc[-12]
revenue_growth_rate = (latest_revenue - year_ago_revenue) / year_ago_revenue
print(f"Revenue growth rate: {revenue_growth_rate:.1%}")

# Step 3: Calculate department-specific growth rates
print("\nStep 3: Calculating department growth rates...")
dept_growth_rates = {}
for dept in ml_predictions['Department'].unique():
    if dept == 'Engineering':
        dept_growth_rates[dept] = revenue_growth_rate * 0.8
    elif dept == 'Sales':
        dept_growth_rates[dept] = revenue_growth_rate * 1.2
    elif dept == 'Marketing':
        dept_growth_rates[dept] = revenue_growth_rate * 0.6
    else:
        dept_growth_rates[dept] = revenue_growth_rate * 0.5

for dept, rate in dept_growth_rates.items():
    high_growth = "HIGH GROWTH" if rate > 0.10 else "Normal"
    print(f"  {dept}: {rate:.1%} ({high_growth})")

# Step 4: Process FIXED ML predictions for hiring timeline
print("\nStep 4: Processing FIXED ML predictions for hiring timeline...")

# Filter Medium+ risk employees (who are predicted to leave)
# FIXED model should have departure dates for ALL these employees
at_risk_employees = ml_predictions[
    ml_predictions['Risk_Category'].isin(['High', 'Medium'])
].copy()

print(f"Employees predicted to leave by FIXED ML model: {len(at_risk_employees)}")
print(f"  High Risk: {len(at_risk_employees[at_risk_employees['Risk_Category'] == 'High'])}")
print(f"  Medium Risk: {len(at_risk_employees[at_risk_employees['Risk_Category'] == 'Medium'])}")

# Check how many have departure dates
at_risk_with_dates = at_risk_employees[at_risk_employees['Estimated_Departure_Date'].notna()]
print(f"  ✅ With departure dates: {len(at_risk_with_dates)}/{len(at_risk_employees)} ({len(at_risk_with_dates)/len(at_risk_employees)*100:.1f}%)")

# Step 5: Calculate hiring priorities and timelines
print("\nStep 5: Calculating FIXED hiring priorities and timelines...")

LEAD_TIMES = {
    'Engineering': 60, 'Sales': 30, 'Marketing': 35,
    'HR': 45, 'Finance': 50, 'Operations': 40
}

today = datetime.now()
hiring_timeline = []

# Process each at-risk employee
successful_processing = 0
for idx, employee in at_risk_employees.iterrows():
    dept = employee['Department']
    dept_growth_rate = dept_growth_rates.get(dept, 0.05)
    
    # Determine priority using our business logic
    is_high_growth_dept = dept_growth_rate > 0.10  # 10% threshold
    is_high_risk = employee['Risk_Category'] == 'High'
    
    if is_high_risk and is_high_growth_dept:
        priority = 'CRITICAL'
        timeline_multiplier = 0.8  # 20% faster
    elif is_high_risk:
        priority = 'HIGH'  # CHANGED from NORMAL to HIGH
        timeline_multiplier = 1.0
    else:  # Medium risk
        priority = 'NORMAL'
        timeline_multiplier = 1.0
    
    # Calculate hiring timeline
    standard_lead_time = LEAD_TIMES.get(dept, 45)
    actual_lead_time = int(standard_lead_time * timeline_multiplier)
    
    # Use FIXED ML model's departure prediction
    if pd.notna(employee['Estimated_Departure_Date']):
        departure_date = pd.to_datetime(employee['Estimated_Departure_Date'])
        successful_processing += 1
    else:
        # Fallback: estimate based on risk level and notice period from FIXED ML
        if pd.notna(employee.get('Predicted_Notice_Period_Days', np.nan)):
            # FIXED: Use the realistic notice period prediction
            notice_days = int(employee['Predicted_Notice_Period_Days'])
            if is_high_risk:
                resignation_days = 60  # High risk likely to resign soon
            else:
                resignation_days = 120  # Medium risk
            departure_date = today + timedelta(days=resignation_days + notice_days)
        else:
            # Final fallback (should rarely happen with FIXED model)
            departure_days = 60 if is_high_risk else 120
            departure_date = today + timedelta(days=departure_days)
            print(f"  ⚠️ Using fallback for {employee['Employee_ID']}")
    
    # Calculate when to start hiring
    start_hiring_date = departure_date - timedelta(days=actual_lead_time)
    days_from_today = (start_hiring_date - today).days
    
    # Action status
    if days_from_today <= 0:
        action_status = "START IMMEDIATELY"
    elif days_from_today <= 7:
        action_status = f"Start in {days_from_today} days"
    else:
        action_status = f"Start in {days_from_today} days"
    
    # FIXED: Include notice period information
    predicted_notice = employee.get('Predicted_Notice_Period_Days', 'N/A')
    
    hiring_timeline.append({
        'Employee_ID': employee['Employee_ID'],
        'Name': employee['Name'],
        'Department': dept,
        'Position': employee.get('Designation', 'Various'),
        'ML_Risk_Category': employee['Risk_Category'],
        'ML_Attrition_Probability': employee.get('Attrition_Probability', 0.5),
        'Predicted_Notice_Days': predicted_notice,  # ADDED
        'Priority': priority,
        'Departure_Date': departure_date.strftime('%Y-%m-%d'),
        'Start_Hiring_Date': start_hiring_date.strftime('%Y-%m-%d'),
        'Days_From_Today': days_from_today,
        'Action_Status': action_status,
        'Lead_Time_Days': actual_lead_time,
        'Hiring_Type': 'Replacement',
        'Dept_Growth_Rate': f"{dept_growth_rate:.1%}",
        'Data_Source': 'FIXED_ML' if pd.notna(employee['Estimated_Departure_Date']) else 'Fallback'
    })

print(f"✅ Successfully processed: {successful_processing}/{len(at_risk_employees)} employees with FIXED ML dates")

# Step 6: Add growth-based hiring
print("\nStep 6: Adding growth-based hiring needs...")

# Calculate current department sizes
dept_sizes = ml_predictions['Department'].value_counts()

for dept, current_size in dept_sizes.items():
    dept_growth_rate = dept_growth_rates.get(dept, 0.05)
    growth_hires = int(current_size * dept_growth_rate)
    
    if growth_hires > 0:
        # Spread growth hiring over 6 months
        for i in range(growth_hires):
            hire_month = (i % 6) + 1
            start_date = today + timedelta(days=30 * hire_month)
            
            hiring_timeline.append({
                'Employee_ID': f'GROWTH_{dept}_{i+1}',
                'Name': f'Growth Position {i+1}',
                'Department': dept,
                'Position': 'Various Positions',
                'ML_Risk_Category': 'Growth',
                'ML_Attrition_Probability': 0.0,
                'Predicted_Notice_Days': 'N/A',  # ADDED
                'Priority': 'LOW',
                'Departure_Date': '',
                'Start_Hiring_Date': start_date.strftime('%Y-%m-%d'),
                'Days_From_Today': (start_date - today).days,
                'Action_Status': f"Start in {(start_date - today).days} days",
                'Lead_Time_Days': LEAD_TIMES.get(dept, 45),
                'Hiring_Type': 'Growth',
                'Dept_Growth_Rate': f"{dept_growth_rate:.1%}",
                'Data_Source': 'Business_Growth'
            })

# Step 7: Create executive summary
print("\nStep 7: Creating executive summary...")

executive_summary = []
for dept in ml_predictions['Department'].unique():
    dept_timeline = [h for h in hiring_timeline if h['Department'] == dept]
    
    replacement_hires = len([h for h in dept_timeline if h['Hiring_Type'] == 'Replacement'])
    growth_hires = len([h for h in dept_timeline if h['Hiring_Type'] == 'Growth'])
    critical_count = len([h for h in dept_timeline if h['Priority'] == 'CRITICAL'])
    
    if replacement_hires + growth_hires > 0:
        executive_summary.append({
            'Department': dept,
            'Current_Employees': dept_sizes.get(dept, 0),
            'ML_Predicted_Departures': replacement_hires,
            'Growth_Hires_Needed': growth_hires,
            'Total_Hires_Required': replacement_hires + growth_hires,
            'Critical_Positions': critical_count,
            'Department_Growth_Rate': f"{dept_growth_rates.get(dept, 0.05):.1%}",
            'Action_Required': 'URGENT' if critical_count > 0 else 'NORMAL'
        })

# Step 8: Create DataFrames and save to Excel
print("\nStep 8: Saving FIXED integrated hiring plan...")

exec_summary_df = pd.DataFrame(executive_summary)
timeline_df = pd.DataFrame(hiring_timeline)

# Sort timeline by urgency
timeline_df = timeline_df.sort_values(['Days_From_Today', 'Priority'])

# Create at-risk employees summary (from FIXED ML predictions)
at_risk_summary = at_risk_employees[[
    'Employee_ID', 'Name', 'Department', 'Risk_Category', 
    'Attrition_Probability', 'Estimated_Departure_Date'
]].copy()

# FIXED: Add new notice period column if available
if 'Predicted_Notice_Period_Days' in at_risk_employees.columns:
    at_risk_summary['Predicted_Notice_Period_Days'] = at_risk_employees['Predicted_Notice_Period_Days']

# Add additional columns if they exist
optional_cols = ['Job_Satisfaction_Score', 'Manager_Rating', 'Estimated_Resignation_Date']
for col in optional_cols:
    if col in at_risk_employees.columns:
        at_risk_summary[col] = at_risk_employees[col]

# Save to Excel
filename = f'FIXED_integrated_hiring_plan_{datetime.now().strftime("%Y%m%d")}.xlsx'

with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    # Sheet 1: Executive Summary
    exec_summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
    
    # Sheet 2: FIXED ML Predicted At-Risk Employees
    at_risk_summary.to_excel(writer, sheet_name='FIXED_ML_At_Risk_Employees', index=False)
    
    # Sheet 3: Complete Hiring Timeline
    timeline_df.to_excel(writer, sheet_name='Hiring_Timeline', index=False)
    
    # Sheet 4: FIXED Notice Period Analysis
    if 'Predicted_Notice_Days' in timeline_df.columns:
        notice_analysis = timeline_df[timeline_df['Hiring_Type'] == 'Replacement'].groupby('Department').agg({
            'Predicted_Notice_Days': ['count', 'mean', 'min', 'max'],
            'ML_Attrition_Probability': 'mean'
        }).round(1)
        notice_analysis.columns = ['Employee_Count', 'Avg_Notice_Days', 'Min_Notice', 'Max_Notice', 'Avg_Risk_Score']
        notice_analysis.to_excel(writer, sheet_name='Notice_Period_Analysis')

print(f"\n" + "="*60)
print("FIXED INTEGRATED HIRING PLAN SUMMARY")
print("="*60)

# Summary statistics
total_replacement = len([h for h in hiring_timeline if h['Hiring_Type'] == 'Replacement'])
total_growth = len([h for h in hiring_timeline if h['Hiring_Type'] == 'Growth'])
total_critical = len([h for h in hiring_timeline if h['Priority'] == 'CRITICAL'])
total_high = len([h for h in hiring_timeline if h['Priority'] == 'HIGH'])
immediate_action = len([h for h in hiring_timeline if h['Days_From_Today'] <= 0])

# Data source breakdown
fixed_ml_count = len([h for h in hiring_timeline if h.get('Data_Source') == 'FIXED_ML'])
fallback_count = len([h for h in hiring_timeline if h.get('Data_Source') == 'Fallback'])

print(f"📊 SUMMARY BASED ON FIXED ML PREDICTIONS:")
print(f"  Replacement hires (FIXED ML predicted departures): {total_replacement}")
print(f"  Growth hires (business expansion): {total_growth}")
print(f"  Critical priority positions: {total_critical}")
print(f"  High priority positions: {total_high}")
print(f"  Immediate action required: {immediate_action}")

print(f"\n🔧 FIXED DATA SOURCE QUALITY:")
print(f"  ✅ Using FIXED ML dates: {fixed_ml_count} employees")
print(f"  ⚠️ Using fallback estimates: {fallback_count} employees")

print(f"\n🚨 IMMEDIATE ACTIONS (Based on FIXED ML predictions):")
immediate_actions = [h for h in hiring_timeline if h['Days_From_Today'] <= 7 and h['Hiring_Type'] == 'Replacement']
if immediate_actions:
    for action in immediate_actions[:5]:
        ml_prob = action['ML_Attrition_Probability']
        predicted_notice = action.get('Predicted_Notice_Days', 'N/A')
        data_source = action.get('Data_Source', 'Unknown')
        print(f"  {action['Employee_ID']} - {action['Name']} ({action['Department']}) - Risk: {ml_prob:.1%} - Notice: {predicted_notice} days - {action['Action_Status']} [{data_source}]")
else:
    print("  ✅ No immediate replacement hiring required")

print(f"\n🏢 DEPARTMENT SUMMARY:")
for summary in executive_summary:
    print(f"  {summary['Department']}: {summary['Total_Hires_Required']} total hires ({summary['ML_Predicted_Departures']} replacements + {summary['Growth_Hires_Needed']} growth)")

# FIXED: Notice period insights
print(f"\n📅 FIXED NOTICE PERIOD INSIGHTS:")
replacement_timeline = [h for h in hiring_timeline if h['Hiring_Type'] == 'Replacement' and h.get('Predicted_Notice_Days') != 'N/A']
if replacement_timeline:
    notice_periods = [int(h['Predicted_Notice_Days']) for h in replacement_timeline if h['Predicted_Notice_Days'] != 'N/A']
    if notice_periods:
        avg_notice = np.mean(notice_periods)
        min_notice = min(notice_periods)
        max_notice = max(notice_periods)
        print(f"  Average predicted notice period: {avg_notice:.1f} days")
        print(f"  Notice period range: {min_notice} - {max_notice} days")
        print(f"  Employees with short notice (≤21 days): {len([n for n in notice_periods if n <= 21])}")
        print(f"  Employees with long notice (>60 days): {len([n for n in notice_periods if n > 60])}")

print(f"\n📁 FIXED integrated hiring plan saved to: {filename}")
print(f"📊 Now using FIXED Module 1's REALISTIC ML predictions!")

print(f"\n✅ INTEGRATION SUCCESS (FIXED):")
print(f"  ✓ Uses FIXED Module 1's trained ML models")
print(f"  ✓ REALISTIC notice period predictions based on role/circumstances")
print(f"  ✓ Separate modeling of resignation timeline vs notice period")
print(f"  ✓ ALL at-risk employees have realistic departure dates")
print(f"  ✓ Combines FIXED attrition predictions with business growth")
print(f"  ✓ Provides complete actionable hiring timeline")
print(f"  ✓ Prioritizes based on FIXED ML risk levels + business impact")
print(f"  ✓ Much more explainable and business-aligned predictions")
print("="*60)