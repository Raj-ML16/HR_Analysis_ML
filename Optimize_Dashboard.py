#!/usr/bin/env python3
"""
HR Hiring Timeline Optimizer Dashboard
Sub-module 3: Optimize hiring timelines considering lead time required for onboarding to avoid resource gaps

Run with: streamlit run hr_hiring_timeline_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the optimizer
try:
    import sys
    import os
    # Add current directory to path for import
    sys.path.append(os.getcwd())
    
    # Try to import from the file (assuming it's named Optimize_calculator.py)
    from Optimize_calculator import DataDrivenHiringOptimizer
    OPTIMIZER_AVAILABLE = True
    st.success("✅ HR Timeline Optimizer loaded successfully")
except ImportError as e:
    st.error(f"❌ Could not import DataDrivenHiringOptimizer: {str(e)}")
    st.write("Please ensure Optimize_calculator.py is in the same directory")
    OPTIMIZER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="HR Timeline Optimizer",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load optimization data
@st.cache_data
def load_hiring_optimization():
    """Load hiring timeline optimization results"""
    if not OPTIMIZER_AVAILABLE:
        return None, None
    
    try:
        optimizer = DataDrivenHiringOptimizer()
        results_df = optimizer.run_optimization()
        filename = optimizer.export_results(results_df)
        return results_df, filename
    except Exception as e:
        st.error(f"Error running optimization: {str(e)}")
        return None, None

# Helper functions for HR-focused analytics
def categorize_urgency(days_until_action):
    """Categorize hiring urgency for HR"""
    if days_until_action <= 0:
        return "🚨 Start Immediately"
    elif days_until_action <= 7:
        return "📅 Start This Week"
    elif days_until_action <= 30:
        return "📋 Start This Month"
    else:
        return "⏳ Future Planning"

def get_risk_level(ml_probability):
    """Convert ML probability to HR risk levels"""
    if pd.isna(ml_probability) or ml_probability == 'N/A':
        return "Unknown"
    try:
        prob = float(ml_probability)
        if prob >= 0.7:
            return "High Risk"
        elif prob >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"
    except:
        return "Unknown"

def main():
    st.title("⏰ HR Hiring Timeline Optimizer")
    st.markdown("**Sub-module 3: Eliminate resource gaps through optimized hiring timelines**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 HR Dashboard Controls")
        
        # Key information for HR
        with st.expander("ℹ️ What This Tool Does"):
            st.markdown("""
            **Objective:** Optimize hiring timelines considering lead time required for onboarding to avoid resource gaps
            
            **Key Benefits:**
            - ✅ Zero resource gaps achieved
            - 📅 Optimal hiring start dates calculated
            - 💰 Cost-effective strategies selected
            - 🎯 Priority-based recommendations
            
            **For HR Team:**
            - Know exactly when to start hiring
            - Understand which positions are most urgent
            - See predicted employee departure timelines
            - Get cost estimates for different strategies
            """)
        
        # Refresh analysis
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()

    if not OPTIMIZER_AVAILABLE:
        st.stop()

    # Load data
    with st.spinner("Analyzing hiring timelines and resource gaps..."):
        results_df, export_filename = load_hiring_optimization()

    if results_df is None:
        st.error("Failed to load hiring optimization data")
        return

    # Success message
    st.success(f"✅ Analysis complete! Detailed report exported to: {export_filename}")

    # === KEY HR METRICS ===
    st.subheader("📊 Key HR Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_positions = len(results_df)
        st.metric("Total Positions to Fill", total_positions, help="Employees requiring replacement")
    
    with col2:
        immediate_actions = len(results_df[results_df['Days_Until_Action'] <= 0])
        st.metric("Start Hiring NOW", immediate_actions, 
                 delta="Immediate action required", 
                 delta_color="inverse" if immediate_actions > 0 else "normal")
    
    with col3:
        this_week = len(results_df[results_df['Days_Until_Action'].between(1, 7)])
        st.metric("Start This Week", this_week, 
                 delta="Within 7 days",
                 delta_color="inverse" if this_week > 0 else "normal")
    
    with col4:
        zero_gaps = len(results_df[results_df['Resource_Gap_Days'] == 0])
        st.metric("Zero Resource Gaps", f"{zero_gaps} ({zero_gaps/len(results_df)*100:.0f}%)", 
                 delta="Optimized positions")

    # === IMMEDIATE ACTIONS ALERT ===
    immediate_df = results_df[results_df['Days_Until_Action'] <= 0]
    if len(immediate_df) > 0:
        st.error("🚨 IMMEDIATE HIRING ACTIONS REQUIRED!")
        
        for _, row in immediate_df.iterrows():
            with st.expander(f"🔴 {row['Name']} - {row['Department']} - {row['Designation']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Employee Details:**")
                    st.write(f"• Departure Date: {row['Departure_Date']}")
                    st.write(f"• Risk Level: {get_risk_level(row['ML_Attrition_Probability'])}")
                    if row['ML_Attrition_Probability'] != 'N/A':
                        st.write(f"• Attrition Probability: {row['ML_Attrition_Probability']}")
                
                with col2:
                    st.write("**Hiring Timeline:**")
                    st.write(f"• Should have started: {row['Optimized_Hiring_Start']}")
                    st.write(f"• Days overdue: {abs(row['Days_Until_Action'])}")
                    st.write(f"• New hire ready by: {row['New_Employee_Ready']}")
                
                with col3:
                    st.write("**Strategy & Cost:**")
                    st.write(f"• Strategy: {row['Selected_Strategy']}")
                    st.write(f"• Cost: ${row['Strategy_Cost']:,.2f}")
                    st.write(f"• Resource gap: {row['Resource_Gap_Days']} days")

    # === HR ANALYTICS CHARTS ===
    st.subheader("📈 HR Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hiring urgency distribution
        results_df['Urgency_Category'] = results_df['Days_Until_Action'].apply(categorize_urgency)
        urgency_counts = results_df['Urgency_Category'].value_counts()
        
        fig_urgency = px.pie(
            values=urgency_counts.values,
            names=urgency_counts.index,
            title="Hiring Urgency Distribution",
            color_discrete_sequence=['#ff4444', '#ff8800', '#ffaa00', '#44aa44']
        )
        fig_urgency.update_layout(height=400)
        st.plotly_chart(fig_urgency, use_container_width=True)
    
    with col2:
        # Department workload
        dept_summary = results_df.groupby('Department').agg({
            'Employee_ID': 'count',
            'Strategy_Cost': 'sum'
        }).reset_index()
        dept_summary.columns = ['Department', 'Positions_to_Fill', 'Total_Cost']
        
        fig_dept = px.bar(
            dept_summary,
            x='Department',
            y='Positions_to_Fill',
            color='Total_Cost',
            title="Department Hiring Workload",
            labels={'Positions_to_Fill': 'Positions to Fill', 'Total_Cost': 'Total Cost ($)'},
            color_continuous_scale='Blues'
        )
        fig_dept.update_layout(height=400)
        st.plotly_chart(fig_dept, use_container_width=True)

    # === HIRING TIMELINE VISUALIZATION ===
    st.subheader("📅 Hiring Timeline Roadmap")
    
    # Create timeline data for visualization
    timeline_data = []
    for _, row in results_df.head(10).iterrows():  # Show top 10 for clarity
        # Start hiring date
        timeline_data.append({
            'Employee': f"{row['Name']} ({row['Department']})",
            'Date': pd.to_datetime(row['Optimized_Hiring_Start']),
            'Event': 'Start Hiring Process',
            'Urgency': categorize_urgency(row['Days_Until_Action']),
            'Cost': row['Strategy_Cost']
        })
        
        # New hire ready date
        timeline_data.append({
            'Employee': f"{row['Name']} ({row['Department']})",
            'Date': pd.to_datetime(row['New_Employee_Ready']),
            'Event': 'New Hire Ready',
            'Urgency': categorize_urgency(row['Days_Until_Action']),
            'Cost': row['Strategy_Cost']
        })
        
        # Employee departure date
        timeline_data.append({
            'Employee': f"{row['Name']} ({row['Department']})",
            'Date': pd.to_datetime(row['Departure_Date']),
            'Event': 'Employee Departure',
            'Urgency': categorize_urgency(row['Days_Until_Action']),
            'Cost': row['Strategy_Cost']
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig_timeline = px.scatter(
        timeline_df,
        x='Date',
        y='Employee',
        color='Event',
        symbol='Urgency',
        size='Cost',
        title="Hiring Timeline Roadmap (Top 10 Positions)",
        height=600,
        color_discrete_map={
            'Start Hiring Process': '#2196f3',
            'New Hire Ready': '#4caf50',
            'Employee Departure': '#ff5722'
        }
    )
    fig_timeline.update_layout(showlegend=True)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # === DETAILED HR TABLE ===
    st.subheader("📋 Detailed Hiring Plan")
    
    # Filters for HR
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dept_filter = st.selectbox(
            "Filter by Department:",
            ['All Departments'] + list(results_df['Department'].unique())
        )
    
    with col2:
        urgency_filter = st.selectbox(
            "Filter by Urgency:",
            ['All Urgencies', 'Start Immediately', 'Start This Week', 'Start This Month', 'Future Planning']
        )
    
    with col3:
        strategy_filter = st.selectbox(
            "Filter by Strategy:",
            ['All Strategies'] + list(results_df['Selected_Strategy'].unique())
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if dept_filter != 'All Departments':
        filtered_df = filtered_df[filtered_df['Department'] == dept_filter]
    
    if urgency_filter != 'All Urgencies':
        urgency_map = {
            'Start Immediately': lambda x: x <= 0,
            'Start This Week': lambda x: 1 <= x <= 7,
            'Start This Month': lambda x: 8 <= x <= 30,
            'Future Planning': lambda x: x > 30
        }
        if urgency_filter in urgency_map:
            mask = urgency_map[urgency_filter](filtered_df['Days_Until_Action'])
            filtered_df = filtered_df[mask]
    
    if strategy_filter != 'All Strategies':
        filtered_df = filtered_df[filtered_df['Selected_Strategy'] == strategy_filter]
    
    # Display HR-focused columns
    hr_columns = [
        'Name', 'Department', 'Designation', 'Departure_Date',
        'Optimized_Hiring_Start', 'Days_Until_Action', 'Selected_Strategy',
        'Strategy_Cost', 'Resource_Gap_Days', 'Priority_Level'
    ]
    
    # Add ML columns if available
    if 'ML_Attrition_Probability' in filtered_df.columns:
        hr_columns.extend(['ML_Attrition_Probability', 'ML_Notice_Period_Days'])
    
    # Sort by urgency (most urgent first)
    display_df = filtered_df[hr_columns].sort_values('Days_Until_Action')
    
    # Color coding function
    def highlight_urgency(row):
        if row['Days_Until_Action'] <= 0:
            return ['background-color: #ffebee'] * len(row)  # Light red
        elif row['Days_Until_Action'] <= 7:
            return ['background-color: #fff3e0'] * len(row)  # Light orange
        else:
            return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_urgency, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download filtered data
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"hr_hiring_plan_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # === HR SUMMARY INSIGHTS ===
    st.subheader("💡 HR Summary & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Hiring Impact Analysis**")
        total_investment = results_df['Strategy_Cost'].sum()
        avg_cost_per_hire = results_df['Strategy_Cost'].mean()
        total_gap_eliminated = results_df['Gap_Eliminated'].sum()
        
        st.write(f"• Total hiring investment: ${total_investment:,.2f}")
        st.write(f"• Average cost per position: ${avg_cost_per_hire:,.2f}")
        st.write(f"• Resource gap days eliminated: {total_gap_eliminated}")
        st.write(f"• Zero-gap achievement: {zero_gaps}/{len(results_df)} positions")
    
    with col2:
        st.markdown("**⏰ Action Timeline Summary**")
        immediate = len(results_df[results_df['Days_Until_Action'] <= 0])
        this_week = len(results_df[results_df['Days_Until_Action'].between(1, 7)])
        this_month = len(results_df[results_df['Days_Until_Action'].between(8, 30)])
        future = len(results_df[results_df['Days_Until_Action'] > 30])
        
        st.write(f"• Start immediately: {immediate} positions")
        st.write(f"• Start this week: {this_week} positions")
        st.write(f"• Start this month: {this_month} positions")
        st.write(f"• Future planning: {future} positions")
    
    # Strategy effectiveness
    st.markdown("**🎯 Strategy Distribution & Effectiveness**")
    strategy_summary = results_df.groupby('Selected_Strategy').agg({
        'Employee_ID': 'count',
        'Strategy_Cost': 'mean',
        'Resource_Gap_Days': 'mean'
    }).round(2)
    strategy_summary.columns = ['Positions', 'Avg Cost', 'Avg Gap Days']
    st.dataframe(strategy_summary, use_container_width=True)
    
    # ML Integration status
    if 'ML_Attrition_Probability' in results_df.columns:
        ml_available = len(results_df[results_df['ML_Attrition_Probability'] != 'N/A'])
        st.info(f"🤖 ML Predictions: {ml_available}/{len(results_df)} positions have ML-enhanced predictions")

if __name__ == "__main__":
    main()