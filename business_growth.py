import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SUB-MODULE 4: FINANCIAL GROWTH PREDICTOR")
print("Simple & Effective Financial-Based Hiring Optimization")
print("="*60)

# Step 1: Load Data
print("\nStep 1: Loading financial and business data...")
financial_df = pd.read_csv('financial_data.csv')
business_df = pd.read_csv('business_data.csv')
employee_df = pd.read_csv('employee_data_processed.csv')

financial_df['Month'] = pd.to_datetime(financial_df['Month'])
business_df['Month'] = pd.to_datetime(business_df['Month'])

print(f"✅ Financial data: {len(financial_df)} months")
print(f"✅ Business data: {len(business_df)} months")
print(f"✅ Employee data: {len(employee_df)} employees")

# Step 2: Financial Health Analysis
print("\nStep 2: Financial Health Analysis...")

def calculate_financial_health_score():
    """Calculate simple financial health score"""
    recent_data = financial_df.tail(6)  # Last 6 months
    
    # Key indicators (0-100 scale)
    cash_flow_trend = 1 if recent_data['Cash_Flow'].iloc[-1] > recent_data['Cash_Flow'].iloc[0] else 0
    debt_health = 1 if recent_data['Debt_to_Equity_Ratio'].mean() < 0.6 else 0
    profitability = 1 if recent_data['ROE'].mean() > 0.1 else 0
    
    health_score = (cash_flow_trend + debt_health + profitability) / 3 * 100
    
    if health_score >= 67: risk_level = "Low"
    elif health_score >= 33: risk_level = "Medium" 
    else: risk_level = "High"
    
    return health_score, risk_level

health_score, risk_level = calculate_financial_health_score()
print(f"Financial Health Score: {health_score:.0f}/100 ({risk_level} Risk)")

# Step 3: Enhanced Growth Prediction
print("\nStep 3: Enhanced Growth Prediction...")

def calculate_enhanced_growth_rate():
    """Multi-factor growth prediction"""
    recent_financial = financial_df.tail(12)
    recent_business = business_df.tail(12)
    
    # Factor 1: Revenue trend (40% weight)
    revenue_growth = (recent_business['Revenue'].iloc[-1] - recent_business['Revenue'].iloc[0]) / recent_business['Revenue'].iloc[0]
    
    # Factor 2: Cash flow trend (30% weight) 
    cash_flow_growth = (recent_financial['Cash_Flow'].iloc[-1] - recent_financial['Cash_Flow'].iloc[0]) / abs(recent_financial['Cash_Flow'].iloc[0])
    
    # Factor 3: Investment trend (20% weight)
    rd_growth = (recent_financial['RD_Investment'].iloc[-1] - recent_financial['RD_Investment'].iloc[0]) / recent_financial['RD_Investment'].iloc[0]
    
    # Factor 4: Profitability (10% weight)
    roe_trend = recent_financial['ROE'].iloc[-1] - recent_financial['ROE'].iloc[0]
    
    # Weighted growth score
    enhanced_growth = (
        revenue_growth * 0.4 + 
        cash_flow_growth * 0.3 + 
        rd_growth * 0.2 + 
        roe_trend * 0.1
    )
    
    # Risk adjustment
    risk_multiplier = {"Low": 1.0, "Medium": 0.85, "High": 0.7}
    adjusted_growth = enhanced_growth * risk_multiplier[risk_level]
    
    return revenue_growth, enhanced_growth, adjusted_growth

simple_growth, enhanced_growth, final_growth = calculate_enhanced_growth_rate()

print(f"Simple Revenue Growth: {simple_growth:.1%}")
print(f"Enhanced Multi-Factor Growth: {enhanced_growth:.1%}")
print(f"Risk-Adjusted Growth: {final_growth:.1%}")

# Step 4: Smart Department Priorities
print("\nStep 4: Smart Department Priorities...")

def calculate_department_multipliers():
    """Financial-based department multipliers"""
    recent_data = financial_df.tail(6)
    
    # Base multipliers
    multipliers = {"Engineering": 1.0, "Sales": 1.0, "Marketing": 1.0, "HR": 1.0, "Finance": 1.0, "Operations": 1.0}
    
    # R&D investment trend → Engineering priority
    rd_trend = (recent_data['RD_Investment'].iloc[-1] - recent_data['RD_Investment'].iloc[0]) / recent_data['RD_Investment'].iloc[0]
    if rd_trend > 0.1: multipliers["Engineering"] = 1.3
    
    # Market/Revenue trend → Sales priority  
    if simple_growth < 0.05: multipliers["Sales"] = 1.4  # Low growth needs more sales
    
    # Profit margin trend → Operations priority
    profit_trend = (business_df['Profit_Margin'].iloc[-1] - business_df['Profit_Margin'].iloc[-6]) 
    if profit_trend < 0: multipliers["Operations"] = 1.2  # Declining profits need operations focus
    
    # High risk → Reduce Marketing
    if risk_level == "High": multipliers["Marketing"] = 0.7
    
    return multipliers

dept_multipliers = calculate_department_multipliers()

print("Department Growth Multipliers:")
for dept, mult in dept_multipliers.items():
    status = "📈 HIGH" if mult > 1.1 else "📉 LOW" if mult < 0.9 else "➡️ NORMAL"
    print(f"  {dept}: {mult:.1f}x {status}")

# Step 5: Budget Capacity Check
print("\nStep 5: Budget Capacity Analysis...")

def calculate_hiring_budget():
    """Estimate hiring budget from cash flow"""
    avg_cash_flow = financial_df['Cash_Flow'].tail(6).mean()
    
    # Assume 15% of positive cash flow can go to hiring
    if avg_cash_flow > 0:
        hiring_budget = avg_cash_flow * 0.15
        budget_status = "✅ GOOD"
    else:
        hiring_budget = 0
        budget_status = "❌ TIGHT"
    
    return hiring_budget, budget_status

hiring_budget, budget_status = calculate_hiring_budget()
print(f"Estimated Hiring Budget: ${hiring_budget:,.0f}/month ({budget_status})")

# Step 6: Generate Enhanced Hiring Recommendations
print("\nStep 6: Enhanced Hiring Recommendations...")

# Get current department sizes
dept_sizes = employee_df[employee_df['Status'] == 'Active']['Department'].value_counts()

# Calculate enhanced hiring needs
enhanced_hiring = []
total_budget_needed = 0

for dept, current_size in dept_sizes.items():
    enhanced_growth_rate = final_growth * dept_multipliers.get(dept, 1.0)
    growth_hires = max(0, int(current_size * enhanced_growth_rate))
    
    # Estimate cost (simplified)
    avg_salary = {"Engineering": 80000, "Sales": 60000, "Marketing": 55000, 
                  "HR": 50000, "Finance": 65000, "Operations": 45000}
    monthly_cost = growth_hires * avg_salary.get(dept, 50000) / 12
    total_budget_needed += monthly_cost
    
    if growth_hires > 0:
        enhanced_hiring.append({
            'Department': dept,
            'Current_Size': current_size,
            'Simple_Growth_Hires': int(current_size * simple_growth),
            'Enhanced_Growth_Hires': growth_hires,
            'Growth_Rate_Used': f"{enhanced_growth_rate:.1%}",
            'Priority': dept_multipliers.get(dept, 1.0),
            'Monthly_Cost': f"${monthly_cost:,.0f}"
        })

# Budget feasibility
budget_feasible = "✅ FEASIBLE" if total_budget_needed <= hiring_budget else "⚠️ OVER BUDGET"

print(f"Total Monthly Cost: ${total_budget_needed:,.0f} ({budget_feasible})")

# Step 7: Export Results
print("\nStep 7: Exporting Enhanced Results...")

results_df = pd.DataFrame(enhanced_hiring)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'Financial_Enhanced_Hiring_Plan_{timestamp}.xlsx'

# Create summary
summary_data = {
    'Metric': [
        'Simple Revenue Growth', 'Enhanced Multi-Factor Growth', 'Risk-Adjusted Growth',
        'Financial Health Score', 'Risk Level', 'Hiring Budget Available',
        'Total Hiring Cost', 'Budget Status'
    ],
    'Value': [
        f"{simple_growth:.1%}", f"{enhanced_growth:.1%}", f"{final_growth:.1%}",
        f"{health_score:.0f}/100", risk_level, f"${hiring_budget:,.0f}",
        f"${total_budget_needed:,.0f}", budget_feasible.split()[1]
    ]
}

with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    # Enhanced hiring plan
    if len(results_df) > 0:
        results_df.to_excel(writer, sheet_name='Enhanced_Hiring_Plan', index=False)
    
    # Financial summary
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Financial_Summary', index=False)
    
    # Department priorities
    priority_df = pd.DataFrame(list(dept_multipliers.items()), columns=['Department', 'Priority_Multiplier'])
    priority_df.to_excel(writer, sheet_name='Department_Priorities', index=False)

print(f"✅ Results saved to: {filename}")

# Final Summary
print(f"\n" + "="*60)
print("FINANCIAL GROWTH PREDICTION SUMMARY")
print("="*60)
print(f"🎯 Enhanced Growth Rate: {final_growth:.1%} (vs {simple_growth:.1%} simple)")
print(f"💰 Financial Health: {health_score:.0f}/100 ({risk_level} Risk)")
print(f"💵 Budget Status: {budget_status}")
print(f"📊 Total Enhanced Hires: {sum([h['Enhanced_Growth_Hires'] for h in enhanced_hiring])}")
print(f"🏢 High Priority Departments: {[d for d, m in dept_multipliers.items() if m > 1.1]}")
print(f"📁 Detailed Results: {filename}")
print("✅ Sub-Module 4 Complete - Financial Intelligence Added!")
print("="*60)