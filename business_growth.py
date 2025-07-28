import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SUB-MODULE 4: ENHANCED FINANCIAL GROWTH PREDICTOR")
print("Smart Financial-Based Hiring Optimization with Department Analysis")
print("="*60)

# Step 1: Load Data
print("\nStep 1: Loading financial and business data...")
try:
    financial_df = pd.read_csv('financial_data.csv')
    business_df = pd.read_csv('business_data.csv')
    employee_df = pd.read_csv('employee_data_processed.csv')

    financial_df['Month'] = pd.to_datetime(financial_df['Month'])
    business_df['Month'] = pd.to_datetime(business_df['Month'])

    print(f"✅ Financial data: {len(financial_df)} months")
    print(f"✅ Business data: {len(business_df)} months")
    print(f"✅ Employee data: {len(employee_df)} employees")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Step 2: Enhanced Financial Health Analysis
print("\nStep 2: Enhanced Financial Health Analysis...")

def calculate_enhanced_financial_health():
    """Enhanced financial health with more indicators"""
    recent_data = financial_df.tail(6)  # Last 6 months
    
    # Indicator 1: Cash Flow Trend (0-1)
    cash_flow_trend = 1 if recent_data['Cash_Flow'].iloc[-1] > recent_data['Cash_Flow'].iloc[0] else 0
    
    # Indicator 2: Debt Health (0-1) - More nuanced
    avg_debt_ratio = recent_data['Debt_to_Equity_Ratio'].mean()
    if avg_debt_ratio < 0.3: debt_health = 1.0
    elif avg_debt_ratio < 0.6: debt_health = 0.7
    elif avg_debt_ratio < 1.0: debt_health = 0.4
    else: debt_health = 0.0
    
    # Indicator 3: Profitability Trend (0-1)
    roe_recent = recent_data['ROE'].mean()
    if roe_recent > 0.15: profitability = 1.0
    elif roe_recent > 0.10: profitability = 0.8
    elif roe_recent > 0.05: profitability = 0.5
    else: profitability = 0.0
    
    # Indicator 4: Working Capital Health (NEW)
    working_capital_trend = 1 if recent_data['Working_Capital'].iloc[-1] > recent_data['Working_Capital'].iloc[0] else 0
    
    # Weighted health score
    health_score = (cash_flow_trend * 0.3 + debt_health * 0.3 + 
                   profitability * 0.3 + working_capital_trend * 0.1) * 100
    
    if health_score >= 75: risk_level = "Low"
    elif health_score >= 50: risk_level = "Medium" 
    else: risk_level = "High"
    
    return health_score, risk_level, {
        'cash_flow_trend': cash_flow_trend,
        'debt_health': debt_health,
        'profitability': profitability,
        'working_capital_trend': working_capital_trend
    }

health_score, risk_level, health_details = calculate_enhanced_financial_health()
print(f"Financial Health Score: {health_score:.0f}/100 ({risk_level} Risk)")
print(f"  Cash Flow: {'✅' if health_details['cash_flow_trend'] else '❌'}")
print(f"  Debt Health: {health_details['debt_health']:.1f}/1.0")
print(f"  Profitability: {health_details['profitability']:.1f}/1.0")
print(f"  Working Capital: {'✅' if health_details['working_capital_trend'] else '❌'}")

# Step 3: Department Performance Analysis (NEW)
print("\nStep 3: Department Performance Analysis...")

def analyze_department_performance():
    """Analyze each department's current performance"""
    dept_performance = {}
    active_employees = employee_df[employee_df['Status'] == 'Active']
    
    for dept in active_employees['Department'].unique():
        dept_data = active_employees[active_employees['Department'] == dept]
        
        # Performance indicators
        avg_performance = dept_data['Performance_Rating'].mean()
        avg_satisfaction = dept_data['Job_Satisfaction_Score'].mean()
        recent_promotions = len(dept_data[dept_data['Days_Since_Last_Promotion'] < 365])
        
        # Performance score (0-1)
        performance_score = (avg_performance / 5.0 * 0.5 + 
                           avg_satisfaction / 10.0 * 0.3 + 
                           min(recent_promotions / len(dept_data), 0.3) * 0.2)
        
        dept_performance[dept] = {
            'performance_score': performance_score,
            'avg_performance': avg_performance,
            'avg_satisfaction': avg_satisfaction,
            'promotion_rate': recent_promotions / len(dept_data)
        }
    
    return dept_performance

dept_performance = analyze_department_performance()
print("Department Performance Analysis:")
for dept, perf in dept_performance.items():
    status = "🏆 EXCELLENT" if perf['performance_score'] > 0.8 else "⚠️ NEEDS HELP" if perf['performance_score'] < 0.6 else "✅ GOOD"
    print(f"  {dept}: {perf['performance_score']:.2f} {status}")

# Step 4: Enhanced Growth Prediction
print("\nStep 4: Enhanced Growth Prediction...")

def calculate_enhanced_growth_rate():
    """Enhanced multi-factor growth prediction"""
    recent_financial = financial_df.tail(12)
    recent_business = business_df.tail(12)
    
    # Factor 1: Revenue trend (35% weight)
    revenue_growth = (recent_business['Revenue'].iloc[-1] - recent_business['Revenue'].iloc[0]) / recent_business['Revenue'].iloc[0]
    
    # Factor 2: Cash flow trend (25% weight) 
    cash_flow_start = recent_financial['Cash_Flow'].iloc[0]
    cash_flow_end = recent_financial['Cash_Flow'].iloc[-1]
    if cash_flow_start != 0:
        cash_flow_growth = (cash_flow_end - cash_flow_start) / abs(cash_flow_start)
    else:
        cash_flow_growth = 0
    
    # Factor 3: Investment trend (25% weight)
    rd_growth = (recent_financial['RD_Investment'].iloc[-1] - recent_financial['RD_Investment'].iloc[0]) / recent_financial['RD_Investment'].iloc[0]
    
    # Factor 4: Market position (10% weight) - NEW
    market_share_trend = (recent_business['Market_Share'].iloc[-1] - recent_business['Market_Share'].iloc[0])
    
    # Factor 5: Efficiency (5% weight) - NEW
    profit_margin_trend = (recent_business['Profit_Margin'].iloc[-1] - recent_business['Profit_Margin'].iloc[0])
    
    # Weighted growth score
    enhanced_growth = (
        revenue_growth * 0.35 + 
        cash_flow_growth * 0.25 + 
        rd_growth * 0.25 + 
        market_share_trend * 0.10 + 
        profit_margin_trend * 0.05
    )
    
    # Risk adjustment
    risk_multiplier = {"Low": 1.0, "Medium": 0.85, "High": 0.7}
    adjusted_growth = enhanced_growth * risk_multiplier[risk_level]
    
    return revenue_growth, enhanced_growth, adjusted_growth

simple_growth, enhanced_growth, final_growth = calculate_enhanced_growth_rate()

print(f"Simple Revenue Growth: {simple_growth:.1%}")
print(f"Enhanced Multi-Factor Growth: {enhanced_growth:.1%}")
print(f"Risk-Adjusted Growth: {final_growth:.1%}")

# Step 5: Smart Department Priorities (ENHANCED)
print("\nStep 5: Smart Department Priorities...")

def calculate_smart_department_multipliers():
    """Enhanced department multipliers with performance consideration"""
    recent_financial = financial_df.tail(6)
    recent_business = business_df.tail(6)
    
    # Base multipliers
    multipliers = {}
    
    for dept in dept_performance.keys():
        base_multiplier = 1.0
        perf = dept_performance[dept]
        
        # Performance-based adjustment
        if perf['performance_score'] < 0.6:
            performance_adj = 1.3  # Poor performance needs more people
        elif perf['performance_score'] > 0.8:
            performance_adj = 0.9  # Excellent performance needs fewer people
        else:
            performance_adj = 1.0
        
        # Financial indicator adjustments
        if dept == "Engineering":
            rd_trend = (recent_financial['RD_Investment'].iloc[-1] - recent_financial['RD_Investment'].iloc[0]) / recent_financial['RD_Investment'].iloc[0]
            financial_adj = 1.4 if rd_trend > 0.15 else 1.2 if rd_trend > 0.05 else 1.0
            
        elif dept == "Sales":
            revenue_trend = simple_growth
            market_trend = recent_business['Market_Share'].iloc[-1] - recent_business['Market_Share'].iloc[0]
            financial_adj = 1.5 if revenue_trend < 0.03 or market_trend < 0 else 1.0
            
        elif dept == "Marketing":
            customer_acq = recent_business['Customer_Acquisition'].iloc[-1] - recent_business['Customer_Acquisition'].iloc[0]
            financial_adj = 0.7 if risk_level == "High" else 1.2 if customer_acq < 0 else 1.0
            
        elif dept == "Operations":
            profit_trend = recent_business['Profit_Margin'].iloc[-1] - recent_business['Profit_Margin'].iloc[-6]
            financial_adj = 1.3 if profit_trend < -0.02 else 1.0
            
        else:  # HR, Finance, etc.
            financial_adj = 0.8 if risk_level == "High" else 1.0
        
        # Combined multiplier
        final_multiplier = base_multiplier * performance_adj * financial_adj
        multipliers[dept] = min(final_multiplier, 2.0)  # Cap at 2x
    
    return multipliers

dept_multipliers = calculate_smart_department_multipliers()

print("Smart Department Growth Multipliers:")
for dept, mult in dept_multipliers.items():
    perf_score = dept_performance[dept]['performance_score']
    status = "🚀 CRITICAL" if mult > 1.4 else "📈 HIGH" if mult > 1.1 else "📉 LOW" if mult < 0.9 else "➡️ NORMAL"
    print(f"  {dept}: {mult:.2f}x {status} (Performance: {perf_score:.2f})")

# Step 6: Enhanced Budget Analysis
print("\nStep 6: Enhanced Budget Analysis...")

def calculate_comprehensive_budget():
    """More sophisticated budget calculation"""
    recent_financial = financial_df.tail(6)
    
    avg_cash_flow = recent_financial['Cash_Flow'].mean()
    avg_working_capital = recent_financial['Working_Capital'].mean()
    
    # Conservative budget calculation
    if avg_cash_flow > 0 and avg_working_capital > 0:
        # Base budget from cash flow
        base_budget = avg_cash_flow * 0.12  # Reduced from 15% to 12% for safety
        
        # Working capital adjustment
        wc_adjustment = min(avg_working_capital * 0.02, base_budget * 0.5)
        
        total_budget = base_budget + wc_adjustment
        budget_status = "✅ STRONG" if total_budget > base_budget * 1.3 else "✅ GOOD"
    else:
        total_budget = max(0, avg_cash_flow * 0.05)  # Very conservative
        budget_status = "⚠️ LIMITED" if total_budget > 0 else "❌ TIGHT"
    
    return total_budget, budget_status

hiring_budget, budget_status = calculate_comprehensive_budget()
print(f"Enhanced Hiring Budget: ${hiring_budget:,.0f}/month ({budget_status})")

# Step 7: Generate Comprehensive Hiring Recommendations
print("\nStep 7: Comprehensive Hiring Recommendations...")

def get_hiring_justification(dept, multiplier, performance):
    """Generate hiring justification"""
    if multiplier > 1.4:
        return f"Critical need: Low performance ({performance['performance_score']:.2f}) + strategic priority"
    elif multiplier > 1.1:
        return f"High priority: Strategic investment or performance improvement needed"
    elif multiplier < 0.9:
        return f"Conservative: Financial constraints or good current performance"
    else:
        return f"Normal growth: Steady performance and balanced priorities"

# Get current department sizes
dept_sizes = employee_df[employee_df['Status'] == 'Active']['Department'].value_counts()

# Calculate comprehensive hiring needs
enhanced_hiring = []
total_budget_needed = 0

for dept, current_size in dept_sizes.items():
    if dept in dept_multipliers:
        enhanced_growth_rate = final_growth * dept_multipliers[dept]
        growth_hires = max(0, int(current_size * enhanced_growth_rate))
        
        # Enhanced cost estimation
        avg_salary = {"Engineering": 85000, "Sales": 65000, "Marketing": 58000, 
                      "HR": 52000, "Finance": 68000, "Operations": 48000}
        
        monthly_cost = growth_hires * avg_salary.get(dept, 55000) / 12
        total_budget_needed += monthly_cost
        
        if growth_hires > 0:
            enhanced_hiring.append({
                'Department': dept,
                'Current_Size': current_size,
                'Performance_Score': f"{dept_performance[dept]['performance_score']:.2f}",
                'Simple_Growth_Hires': int(current_size * simple_growth),
                'Enhanced_Growth_Hires': growth_hires,
                'Growth_Rate_Used': f"{enhanced_growth_rate:.1%}",
                'Priority_Multiplier': f"{dept_multipliers[dept]:.2f}",
                'Monthly_Cost': f"${monthly_cost:,.0f}",
                'Justification': get_hiring_justification(dept, dept_multipliers[dept], dept_performance[dept])
            })

# Budget feasibility with recommendations
budget_feasible = "✅ FEASIBLE" if total_budget_needed <= hiring_budget else "⚠️ OVER BUDGET"
budget_utilization = (total_budget_needed / hiring_budget * 100) if hiring_budget > 0 else 0

print(f"Total Monthly Cost: ${total_budget_needed:,.0f} ({budget_feasible})")
print(f"Budget Utilization: {budget_utilization:.1f}%")

# Step 8: Export Enhanced Results
print("\nStep 8: Exporting Enhanced Results...")

results_df = pd.DataFrame(enhanced_hiring)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'Enhanced_Financial_Hiring_Plan_{timestamp}.xlsx'

# Enhanced summary
summary_data = {
    'Metric': [
        'Simple Revenue Growth', 'Enhanced Multi-Factor Growth', 'Risk-Adjusted Growth',
        'Financial Health Score', 'Risk Level', 'Hiring Budget Available',
        'Total Hiring Cost', 'Budget Utilization', 'Budget Status'
    ],
    'Value': [
        f"{simple_growth:.1%}", f"{enhanced_growth:.1%}", f"{final_growth:.1%}",
        f"{health_score:.0f}/100", risk_level, f"${hiring_budget:,.0f}",
        f"${total_budget_needed:,.0f}", f"{budget_utilization:.1f}%", budget_feasible.split()[1]
    ]
}

with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    # Enhanced hiring plan
    if len(results_df) > 0:
        results_df.to_excel(writer, sheet_name='Enhanced_Hiring_Plan', index=False)
    
    # Financial summary
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Financial_Summary', index=False)
    
    # Department analysis
    dept_analysis = pd.DataFrame([
        {
            'Department': dept,
            'Performance_Score': perf['performance_score'],
            'Avg_Performance_Rating': perf['avg_performance'],
            'Job_Satisfaction': perf['avg_satisfaction'],
            'Promotion_Rate': perf['promotion_rate'],
            'Priority_Multiplier': dept_multipliers.get(dept, 1.0)
        }
        for dept, perf in dept_performance.items()
    ])
    dept_analysis.to_excel(writer, sheet_name='Department_Analysis', index=False)

print(f"✅ Enhanced results saved to: {filename}")

# Final Enhanced Summary
print(f"\n" + "="*60)
print("ENHANCED FINANCIAL GROWTH PREDICTION SUMMARY")
print("="*60)
print(f"🎯 Enhanced Growth Rate: {final_growth:.1%} (vs {simple_growth:.1%} simple)")
print(f"💰 Financial Health: {health_score:.0f}/100 ({risk_level} Risk)")
print(f"💵 Budget Status: {budget_status} ({budget_utilization:.1f}% utilization)")
print(f"📊 Total Enhanced Hires: {sum([h['Enhanced_Growth_Hires'] for h in enhanced_hiring])}")

# Priority departments with reasoning
high_priority = [(d, m) for d, m in dept_multipliers.items() if m > 1.1]
print(f"🏢 High Priority Departments: {[f'{d} ({m:.2f}x)' for d, m in high_priority]}")

print(f"🔍 Key Insights:")
for dept, mult in sorted(dept_multipliers.items(), key=lambda x: x[1], reverse=True)[:3]:
    perf = dept_performance[dept]['performance_score']
    print(f"   {dept}: {mult:.2f}x multiplier (Performance: {perf:.2f})")

print(f"📁 Detailed Results: {filename}")
print("✅ Enhanced Sub-Module 4 Complete - Advanced Financial Intelligence!")
print("="*60)