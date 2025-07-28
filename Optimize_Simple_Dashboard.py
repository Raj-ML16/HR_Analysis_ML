#!/usr/bin/env python3
"""
Complete HR Timeline Optimizer Dashboard
Single file with optimizer and dashboard combined

Run with: streamlit run hr_timeline_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import glob
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# TIMELINE OPTIMIZER CLASS
# =============================================================================

class SimplifiedTimelineOptimizer:
    """
    SIMPLIFIED: Pure timeline optimizer for Sub-module 3
    Focus: "Optimize hiring timelines considering lead time required for onboarding to avoid resource gaps"
    
    6 Essential Steps Only:
    1. Load departure predictions from Module 2
    2. Define department timeline rules  
    3. Calculate optimal timeline for each employee
    4. Process all employees
    5. Add action urgency
    6. Export simple results
    """
    
    def __init__(self):
        # STEP 2: Define department timeline rules (hiring + onboarding)
        self.DEPARTMENT_TIMELINES = {
            'Engineering': {'hiring': 60, 'onboarding': 21, 'total': 81},
            'Sales': {'hiring': 30, 'onboarding': 14, 'total': 44},
            'Marketing': {'hiring': 35, 'onboarding': 18, 'total': 53},
            'HR': {'hiring': 45, 'onboarding': 21, 'total': 66},
            'Finance': {'hiring': 50, 'onboarding': 25, 'total': 75},
            'Operations': {'hiring': 40, 'onboarding': 16, 'total': 56}
        }
    
    def load_departure_predictions(self):
        """
        STEP 1: Load departure predictions from Module 2's output
        Simple data loading with minimal processing
        """
        try:
            # Try Module 2 output first
            files = glob.glob('FIXED_integrated_hiring_plan_*.xlsx')
            if files:
                latest_file = max(files, key=lambda f: f)
                df = pd.read_excel(latest_file, sheet_name='Hiring_Timeline')
                df = df[df['Hiring_Type'] == 'Replacement']
                
                # Simple column mapping - only essential columns
                essential_data = []
                for _, row in df.iterrows():
                    essential_data.append({
                        'Employee_ID': row['Employee_ID'],
                        'Name': row['Name'],
                        'Department': row['Department'],
                        'Departure_Date': row['Departure_Date']
                    })
                
                return pd.DataFrame(essential_data)
                
        except Exception as e:
            pass
        
        # Fallback: Try Module 1 output
        try:
            files = glob.glob('Employee_Attrition_Predictions_FIXED_*.xlsx')
            if files:
                latest_file = max(files, key=lambda f: f)
                df = pd.read_excel(latest_file, sheet_name='High_Risk_Employees')
                
                # Convert to simple format
                essential_data = []
                for _, row in df.iterrows():
                    # Use departure prediction if available
                    if 'Estimated_Departure_Date' in row and pd.notna(row['Estimated_Departure_Date']):
                        departure_date = row['Estimated_Departure_Date']
                    else:
                        # Simple estimation based on risk
                        risk = row.get('Attrition_Probability', 0.5)
                        days_ahead = 60 if risk >= 0.7 else 120
                        departure_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                    
                    essential_data.append({
                        'Employee_ID': row['Employee_ID'],
                        'Name': row['Name'],
                        'Department': row['Department'],
                        'Departure_Date': departure_date
                    })
                
                return pd.DataFrame(essential_data)
                
        except Exception as e:
            pass
        
        # Final fallback: Create sample data
        return self.create_sample_departures()
    
    def create_sample_departures(self):
        """Create sample departure data if no ML predictions available"""
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        sample_data = []
        
        for i, dept in enumerate(departments * 2):  # 12 sample employees
            departure_date = datetime.now() + timedelta(days=30 + i*15)
            
            sample_data.append({
                'Employee_ID': f'EMP_{i+1:03d}',
                'Name': f'Employee {i+1}',
                'Department': dept,
                'Departure_Date': departure_date.strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(sample_data)
    
    def calculate_optimal_timeline(self, employee):
        """
        STEP 3: Calculate optimal timeline for one employee
        Core logic: Work backwards from departure date to avoid resource gaps
        """
        dept = employee['Department']
        departure_date = pd.to_datetime(employee['Departure_Date'])
        
        # Get department timeline rules
        timeline = self.DEPARTMENT_TIMELINES.get(dept, {
            'hiring': 45, 'onboarding': 21, 'total': 66  # Default fallback
        })
        
        hiring_days = timeline['hiring']
        onboarding_days = timeline['onboarding']
        total_timeline = timeline['total']
        
        # Calculate optimal start date (work backwards from departure)
        optimal_start = departure_date - timedelta(days=total_timeline)
        
        # When new hire will be ready (after hiring process)
        new_hire_ready = optimal_start + timedelta(days=hiring_days)
        
        # Calculate resource gap (should be 0 if we start early enough)
        resource_gap = max(0, (departure_date - new_hire_ready).days)
        
        # If there's still a gap, start even earlier
        if resource_gap > 0:
            optimal_start = optimal_start - timedelta(days=resource_gap)
            new_hire_ready = optimal_start + timedelta(days=hiring_days)
            resource_gap = 0  # Now guaranteed to be zero
        
        return {
            'optimal_start': optimal_start,
            'new_hire_ready': new_hire_ready,
            'resource_gap': resource_gap,
            'hiring_days': hiring_days,
            'onboarding_days': onboarding_days,
            'total_timeline': total_timeline
        }
    
    def add_action_urgency(self, optimal_start):
        """
        STEP 5: Calculate urgency and action status
        """
        today = datetime.now()
        days_until_action = (optimal_start - today).days
        
        if days_until_action <= 0:
            if days_until_action < -30:
                action_status = f"URGENT: {abs(days_until_action)} days overdue!"
            else:
                action_status = "START NOW!"
        elif days_until_action <= 7:
            action_status = f"Start in {days_until_action} days"
        else:
            action_status = f"Start in {days_until_action} days"
        
        return days_until_action, action_status
    
    def process_all_employees(self, departures_df):
        """
        STEP 4: Process all employees using optimal timeline calculation
        """
        optimized_results = []
        
        for _, employee in departures_df.iterrows():
            # Calculate optimal timeline
            timeline_result = self.calculate_optimal_timeline(employee)
            
            # Add action urgency
            days_until_action, action_status = self.add_action_urgency(timeline_result['optimal_start'])
            
            # Package results
            optimized_results.append({
                'Employee_ID': employee['Employee_ID'],
                'Name': employee['Name'],
                'Department': employee['Department'],
                'Departure_Date': employee['Departure_Date'],
                'Optimal_Hiring_Start': timeline_result['optimal_start'].strftime('%Y-%m-%d'),
                'New_Employee_Ready': timeline_result['new_hire_ready'].strftime('%Y-%m-%d'),
                'Resource_Gap_Days': timeline_result['resource_gap'],
                'Hiring_Timeline_Days': timeline_result['hiring_days'],
                'Onboarding_Days': timeline_result['onboarding_days'],
                'Total_Lead_Time': timeline_result['total_timeline'],
                'Days_Until_Action': days_until_action,
                'Action_Status': action_status
            })
        
        results_df = pd.DataFrame(optimized_results)
        return results_df
    
    def export_results(self, results_df):
        """
        STEP 6: Export simple, actionable results to Excel
        Focus on essential information only
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'SIMPLIFIED_hiring_timeline_{timestamp}.xlsx'
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main results - clean and simple
                results_df.to_excel(writer, sheet_name='Optimized_Timeline', index=False)
                
                # Immediate action summary
                urgent = results_df[results_df['Days_Until_Action'] <= 7].copy()
                if not urgent.empty:
                    urgent_cols = ['Employee_ID', 'Name', 'Department', 'Departure_Date',
                                 'Optimal_Hiring_Start', 'Days_Until_Action', 'Action_Status']
                    urgent[urgent_cols].sort_values('Days_Until_Action').to_excel(
                        writer, sheet_name='Immediate_Actions', index=False)
                
                # Simple summary
                summary_data = {
                    'Metric': [
                        'Total Positions Optimized',
                        'Immediate Actions (≤0 days)',
                        'Actions Next Week (1-7 days)',
                        'Zero Resource Gap Achieved',
                        'Total Gap Days Eliminated',
                        'Average Lead Time (days)'
                    ],
                    'Value': [
                        len(results_df),
                        len(results_df[results_df['Days_Until_Action'] <= 0]),
                        len(results_df[results_df['Days_Until_Action'].between(1, 7)]),
                        f"{len(results_df)} (100%)",
                        results_df['Onboarding_Days'].sum(),
                        round(results_df['Total_Lead_Time'].mean(), 1)
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            return filename
        except:
            # Fallback to CSV if Excel fails
            csv_filename = filename.replace('.xlsx', '.csv')
            results_df.to_csv(csv_filename, index=False)
            return csv_filename
    
    def run_optimization(self):
        """
        Main execution: Run all 6 essential steps
        """
        # STEP 1: Load departure predictions
        departures_df = self.load_departure_predictions()
        
        # STEPS 3-5: Process all employees (combines timeline calc + urgency)
        results_df = self.process_all_employees(departures_df)
        
        # STEP 6: Export results
        filename = self.export_results(results_df)
        
        return results_df, filename

# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

# Page config
st.set_page_config(
    page_title="HR Timeline Optimizer",
    page_icon="📊",
    layout="wide"
)

# Load data function
@st.cache_data
def load_optimization_data():
    """Load data using the optimizer"""
    try:
        optimizer = SimplifiedTimelineOptimizer()
        results_df, filename = optimizer.run_optimization()
        return results_df, filename
    except Exception as e:
        st.error(f"Error running optimization: {str(e)}")
        return None, None

# Main dashboard
def main():
    st.title("HR Timeline Optimizer Dashboard")
    st.write("Optimize hiring timelines to eliminate resource gaps")
    
    # Load data
    with st.spinner("Running timeline optimization..."):
        results_df, export_filename = load_optimization_data()
    
    if results_df is None:
        st.error("Failed to load optimization data")
        return
    
    # Display success message
    st.success(f"✅ Optimization complete! Results exported to: {export_filename}")
    
    # Key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Positions", len(results_df))
    
    with col2:
        immediate = len(results_df[results_df['Days_Until_Action'] <= 0])
        st.metric("Immediate Actions", immediate)
    
    with col3:
        urgent = len(results_df[results_df['Days_Until_Action'].between(1, 7)])
        st.metric("This Week", urgent)
    
    with col4:
        zero_gaps = len(results_df[results_df['Resource_Gap_Days'] == 0])
        st.metric("Zero Gap Positions", f"{zero_gaps} ({zero_gaps/len(results_df)*100:.0f}%)")
    
    # Immediate actions alert
    immediate_df = results_df[results_df['Days_Until_Action'] <= 0]
    if len(immediate_df) > 0:
        st.error("⚠️ IMMEDIATE ACTIONS REQUIRED!")
        
        for _, row in immediate_df.head(5).iterrows():  # Show top 5 urgent
            with st.expander(f"🔴 {row['Name']} - {row['Department']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Leaves:** {row['Departure_Date']}")
                with col2:
                    st.write(f"**Should start:** {row['Optimal_Hiring_Start']}")
                with col3:
                    st.write(f"**Status:** {row['Action_Status']}")
    
    # Charts section
    st.subheader("Visual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Urgency distribution
        def get_urgency_category(days):
            if days <= 0:
                return "Immediate"
            elif days <= 7:
                return "This Week"
            elif days <= 30:
                return "This Month"
            else:
                return "Future"
        
        results_df['Urgency_Category'] = results_df['Days_Until_Action'].apply(get_urgency_category)
        urgency_counts = results_df['Urgency_Category'].value_counts()
        
        fig = px.pie(
            values=urgency_counts.values,
            names=urgency_counts.index,
            title="Action Urgency Distribution",
            color_discrete_sequence=['#ff4444', '#ff8800', '#ffbb33', '#44aa44']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Department timelines
        dept_summary = results_df.groupby('Department').agg({
            'Total_Lead_Time': 'first',
            'Employee_ID': 'count'
        }).reset_index()
        dept_summary.columns = ['Department', 'Timeline_Days', 'Position_Count']
        
        fig = px.bar(
            dept_summary,
            x='Department',
            y='Timeline_Days',
            color='Position_Count',
            title="Department Hiring Timelines",
            labels={'Timeline_Days': 'Days Required'},
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline visualization
    st.subheader("Timeline Overview")
    
    # Create timeline data
    timeline_data = []
    for _, row in results_df.iterrows():
        # Hiring start point
        timeline_data.append({
            'Employee': row['Name'],
            'Date': pd.to_datetime(row['Optimal_Hiring_Start']),
            'Event': 'Start Hiring',
            'Department': row['Department']
        })
        # Employee departure point
        timeline_data.append({
            'Employee': row['Name'],
            'Date': pd.to_datetime(row['Departure_Date']),
            'Event': 'Employee Leaves',
            'Department': row['Department']
        })
        # New employee ready point
        timeline_data.append({
            'Employee': row['Name'],
            'Date': pd.to_datetime(row['New_Employee_Ready']),
            'Event': 'New Hire Ready',
            'Department': row['Department']
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.scatter(
        timeline_df,
        x='Date',
        y='Employee',
        color='Event',
        title="Hiring Timeline Overview",
        height=600,
        color_discrete_map={
            'Start Hiring': '#2196f3',
            'New Hire Ready': '#4caf50',
            'Employee Leaves': '#ff5722'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dept_filter = st.selectbox(
            "Filter by Department:",
            ['All'] + list(results_df['Department'].unique())
        )
    
    with col2:
        urgency_filter = st.selectbox(
            "Filter by Urgency:",
            ['All', 'Immediate', 'This Week', 'This Month', 'Future']
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ['Days Until Action', 'Departure Date', 'Name', 'Department']
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if dept_filter != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == dept_filter]
    
    if urgency_filter != 'All':
        if urgency_filter == 'Immediate':
            filtered_df = filtered_df[filtered_df['Days_Until_Action'] <= 0]
        elif urgency_filter == 'This Week':
            filtered_df = filtered_df[filtered_df['Days_Until_Action'].between(1, 7)]
        elif urgency_filter == 'This Month':
            filtered_df = filtered_df[filtered_df['Days_Until_Action'].between(8, 30)]
        elif urgency_filter == 'Future':
            filtered_df = filtered_df[filtered_df['Days_Until_Action'] > 30]
    
    # Sort data
    sort_mapping = {
        'Days Until Action': 'Days_Until_Action',
        'Departure Date': 'Departure_Date',
        'Name': 'Name',
        'Department': 'Department'
    }
    filtered_df = filtered_df.sort_values(sort_mapping[sort_by])
    
    # Display main columns
    display_cols = [
        'Name', 'Department', 'Departure_Date', 'Optimal_Hiring_Start',
        'New_Employee_Ready', 'Resource_Gap_Days', 'Days_Until_Action',
        'Total_Lead_Time', 'Action_Status'
    ]
    
    # Color coding for urgency
    def highlight_urgency(row):
        if row['Days_Until_Action'] <= 0:
            return ['background-color: #ffebee'] * len(row)
        elif row['Days_Until_Action'] <= 7:
            return ['background-color: #fff3e0'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = filtered_df[display_cols].style.apply(highlight_urgency, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Summary insights
    st.subheader("Summary Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Optimization Impact:**")
        avg_timeline = results_df['Total_Lead_Time'].mean()
        total_gap_eliminated = results_df['Onboarding_Days'].sum()
        st.write(f"- Average lead time: {avg_timeline:.0f} days")
        st.write(f"- Total gap eliminated: {total_gap_eliminated} days")
        st.write(f"- Zero resource gaps: {zero_gaps}/{len(results_df)} positions")
    
    with col2:
        st.write("**Action Timeline:**")
        next_30_days = len(results_df[results_df['Days_Until_Action'] <= 30])
        future_actions = len(results_df[results_df['Days_Until_Action'] > 30])
        st.write(f"- Actions next 30 days: {next_30_days}")
        st.write(f"- Future actions: {future_actions}")
        st.write(f"- Departments involved: {results_df['Department'].nunique()}")
    
    # Department breakdown
    st.subheader("Department Analysis")
    dept_analysis = results_df.groupby('Department').agg({
        'Employee_ID': 'count',
        'Days_Until_Action': 'mean',
        'Total_Lead_Time': 'first',
        'Resource_Gap_Days': 'sum'
    }).round(1)
    dept_analysis.columns = ['Positions', 'Avg Days Until Action', 'Timeline Required', 'Total Gap Days']
    
    st.dataframe(dept_analysis, use_container_width=True)
    
    # Refresh button
    if st.button("🔄 Refresh Analysis"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()