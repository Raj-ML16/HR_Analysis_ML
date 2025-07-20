import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="HR Hiring Planning Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .critical-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #d32f2f;
        margin: 0.5rem 0;
    }
    .urgent-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f57c00;
        margin: 0.5rem 0;
    }
    .normal-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2e7d32;
        margin: 0.5rem 0;
    }
    .growth-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1976d2;
        margin: 0.5rem 0;
    }
    .hiring-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_hiring_data():
    """Load hiring plan data"""
    try:
        # Find the latest hiring plan file
        hiring_files = glob.glob('UPDATED_integrated_hiring_plan_*.xlsx')
        
        if not hiring_files:
            st.error("❌ No hiring plan found. Please run Hiring_calculation.py first.")
            return None, None, None
        
        latest_file = max(hiring_files, key=os.path.getmtime)
        
        # Load all sheets
        executive_summary = pd.read_excel(latest_file, sheet_name='Executive_Summary')
        at_risk_employees = pd.read_excel(latest_file, sheet_name='FIXED_ML_At_Risk_Employees')
        hiring_timeline = pd.read_excel(latest_file, sheet_name='Hiring_Timeline')
        
        # Convert date columns
        hiring_timeline['Start_Hiring_Date'] = pd.to_datetime(hiring_timeline['Start_Hiring_Date'])
        hiring_timeline['Departure_Date'] = pd.to_datetime(hiring_timeline['Departure_Date'], errors='coerce')
        
        return executive_summary, at_risk_employees, hiring_timeline, latest_file
        
    except Exception as e:
        st.error(f"Error loading hiring data: {str(e)}")
        return None, None, None, None

def get_urgency_color(days_from_today):
    """Get color based on urgency"""
    if days_from_today <= 0:
        return "#d32f2f"  # Red
    elif days_from_today <= 7:
        return "#f57c00"  # Orange
    elif days_from_today <= 30:
        return "#fbc02d"  # Yellow
    else:
        return "#2e7d32"  # Green

def create_hiring_timeline_chart(hiring_df):
    """Create interactive hiring timeline visualization"""
    
    # Create timeline buckets
    hiring_df['Timeline_Bucket'] = pd.cut(
        hiring_df['Days_From_Today'],
        bins=[-float('inf'), 0, 7, 30, 90, float('inf')],
        labels=['Overdue/Immediate', '1-7 days', '8-30 days', '31-90 days', '90+ days']
    )
    
    # Count by timeline and hiring type
    timeline_counts = hiring_df.groupby(['Timeline_Bucket', 'Hiring_Type']).size().reset_index(name='Count')
    
    fig = px.bar(
        timeline_counts,
        x='Timeline_Bucket',
        y='Count',
        color='Hiring_Type',
        title="Hiring Timeline Overview",
        color_discrete_map={
            'Replacement': '#d32f2f',
            'Growth': '#1976d2'
        },
        hover_data=['Count']
    )
    
    fig.update_layout(
        title={
            'text': "Hiring Timeline Overview",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Timeline",
        yaxis_title="Number of Positions",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_department_hiring_chart(hiring_df):
    """Create department-wise hiring breakdown"""
    dept_summary = hiring_df.groupby(['Department', 'Hiring_Type']).size().reset_index(name='Count')
    
    fig = px.bar(
        dept_summary,
        x='Department',
        y='Count',
        color='Hiring_Type',
        title="Hiring Needs by Department",
        color_discrete_map={
            'Replacement': '#d32f2f',
            'Growth': '#1976d2'
        }
    )
    
    fig.update_layout(
        title={
            'text': "Hiring Needs by Department",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Department",
        yaxis_title="Number of Positions",
        height=400
    )
    
    return fig

def create_priority_distribution_chart(hiring_df):
    """Create priority distribution chart"""
    priority_counts = hiring_df['Priority'].value_counts()
    
    colors = {
        'CRITICAL': '#d32f2f',
        'NORMAL': '#2e7d32',
        'LOW': '#1976d2'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=priority_counts.index,
        values=priority_counts.values,
        marker_colors=[colors.get(label, '#gray') for label in priority_counts.index],
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': "Hiring Priority Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=400
    )
    
    return fig

def display_hiring_item(item, show_employee_details=True):
    """Display individual hiring item"""
    priority_color = "#d32f2f" if item['Priority'] == 'CRITICAL' else "#f57c00" if item['Priority'] == 'NORMAL' else "#2e7d32"
    
    # Determine card style based on urgency
    if item['Days_From_Today'] <= 0:
        card_class = "critical-card"
        urgency_text = "🚨 IMMEDIATE ACTION REQUIRED"
    elif item['Days_From_Today'] <= 7:
        card_class = "urgent-card"
        urgency_text = "⚠️ URGENT - Within 7 Days"
    elif item['Hiring_Type'] == 'Growth':
        card_class = "growth-card"
        urgency_text = "📈 Growth Position"
    else:
        card_class = "normal-card"
        urgency_text = "📅 Planned"
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if item['Hiring_Type'] == 'Replacement' and show_employee_details:
            st.write(f"**👤 Replacing:** {item['Name']}")
            st.write(f"**Employee ID:** {item['Employee_ID']}")
        else:
            st.write(f"**Position:** {item['Position']}")
            st.write(f"**Position ID:** {item['Employee_ID']}")
        
        st.write(f"**Department:** {item['Department']}")
        st.write(f"**Type:** {item['Hiring_Type']}")
    
    with col2:
        st.write(f"**Priority:** {item['Priority']}")
        st.write(f"**Action:** {item['Action_Status']}")
        st.write(f"**Start Hiring:** {item['Start_Hiring_Date'].strftime('%Y-%m-%d')}")
        
        if item['Hiring_Type'] == 'Replacement' and pd.notna(item['Departure_Date']):
            st.write(f"**Departure Date:** {item['Departure_Date'].strftime('%Y-%m-%d')}")
    
    with col3:
        st.markdown(f'<div style="background-color: {priority_color}; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: center; font-weight: bold;">{urgency_text}</div>', unsafe_allow_html=True)
        
        if item['Hiring_Type'] == 'Replacement':
            st.write(f"**ML Risk:** {item.get('ML_Risk_Category', 'N/A')}")
            if 'ML_Attrition_Probability' in item and pd.notna(item['ML_Attrition_Probability']):
                st.write(f"**Risk Score:** {item['ML_Attrition_Probability']:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Load data
    executive_summary, at_risk_employees, hiring_timeline, source_file = load_hiring_data()
    
    if hiring_timeline is None:
        return
    
    # Header
    st.markdown('<div class="main-header">👥 HR Hiring Planning Dashboard</div>', unsafe_allow_html=True)
    
    # File info
    if source_file:
        st.info(f"📊 Data loaded from: {os.path.basename(source_file)}")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Department filter
    departments = ['All Departments'] + sorted(hiring_timeline['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Filter by Department", departments)
    
    # Hiring type filter
    hiring_types = ['All Types', 'Replacement', 'Growth']
    selected_type = st.sidebar.selectbox("Filter by Hiring Type", hiring_types)
    
    # Priority filter
    priorities = ['All Priorities'] + sorted(hiring_timeline['Priority'].unique().tolist())
    selected_priority = st.sidebar.selectbox("Filter by Priority", priorities)
    
    # Timeline filter
    timeline_options = ['All Timeline', 'Immediate (≤0 days)', 'This Week (1-7 days)', 'This Month (≤30 days)', 'This Quarter (≤90 days)']
    selected_timeline = st.sidebar.selectbox("Filter by Timeline", timeline_options)
    
    # Apply filters
    filtered_df = hiring_timeline.copy()
    
    if selected_dept != 'All Departments':
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    
    if selected_type != 'All Types':
        filtered_df = filtered_df[filtered_df['Hiring_Type'] == selected_type]
    
    if selected_priority != 'All Priorities':
        filtered_df = filtered_df[filtered_df['Priority'] == selected_priority]
    
    if selected_timeline != 'All Timeline':
        if selected_timeline == 'Immediate (≤0 days)':
            filtered_df = filtered_df[filtered_df['Days_From_Today'] <= 0]
        elif selected_timeline == 'This Week (1-7 days)':
            filtered_df = filtered_df[(filtered_df['Days_From_Today'] > 0) & (filtered_df['Days_From_Today'] <= 7)]
        elif selected_timeline == 'This Month (≤30 days)':
            filtered_df = filtered_df[filtered_df['Days_From_Today'] <= 30]
        elif selected_timeline == 'This Quarter (≤90 days)':
            filtered_df = filtered_df[filtered_df['Days_From_Today'] <= 90]
    
    # Key metrics
    st.markdown("### 📊 Hiring Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_positions = len(filtered_df)
    replacement_count = len(filtered_df[filtered_df['Hiring_Type'] == 'Replacement'])
    growth_count = len(filtered_df[filtered_df['Hiring_Type'] == 'Growth'])
    critical_count = len(filtered_df[filtered_df['Priority'] == 'CRITICAL'])
    immediate_count = len(filtered_df[filtered_df['Days_From_Today'] <= 0])
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Positions", total_positions)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Replacements", replacement_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Growth Positions", growth_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Critical Priority", critical_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("🚨 Immediate Action", immediate_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Alert for immediate actions
    if immediate_count > 0:
        st.markdown('<div class="critical-card">', unsafe_allow_html=True)
        st.markdown(f"### 🚨 URGENT: {immediate_count} positions need immediate hiring action!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Immediate Actions", "📅 Hiring Timeline", "🏢 Department View", "👥 Employee Details", "📈 Analytics"
    ])
    
    with tab1:
        st.header("🚨 Immediate Actions Required")
        
        immediate_df = filtered_df[filtered_df['Days_From_Today'] <= 7].sort_values('Days_From_Today')
        
        if len(immediate_df) > 0:
            st.subheader(f"⚡ {len(immediate_df)} positions require action within 7 days")
            
            # Immediate action items
            for idx, item in immediate_df.iterrows():
                display_hiring_item(item)
            
            # Download button
            csv = immediate_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Immediate Actions List",
                data=csv,
                file_name=f'immediate_hiring_actions_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        else:
            st.success("✅ No immediate actions required!")
    
    with tab2:
        st.header("📅 Complete Hiring Timeline")
        
        # Interactive charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_timeline = create_hiring_timeline_chart(filtered_df)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            fig_priority = create_priority_distribution_chart(filtered_df)
            st.plotly_chart(fig_priority, use_container_width=True)
        
        # Timeline table
        st.subheader("📋 Detailed Hiring Timeline")
        
        # Sort by urgency
        timeline_sorted = filtered_df.sort_values(['Days_From_Today', 'Priority'])
        
        st.dataframe(
            timeline_sorted[[
                'Employee_ID', 'Name', 'Department', 'Position', 'Hiring_Type',
                'Priority', 'Days_From_Today', 'Start_Hiring_Date', 'Action_Status'
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Days_From_Today": st.column_config.NumberColumn(
                    "Days Until Start",
                    help="Days until hiring should begin",
                    format="%.0f"
                ),
                "Start_Hiring_Date": st.column_config.DateColumn(
                    "Start Hiring Date",
                    help="When to begin recruitment process"
                )
            }
        )
    
    with tab3:
        st.header("🏢 Department-Wise Hiring Plan")
        
        if executive_summary is not None:
            # Department overview chart
            fig_dept = create_department_hiring_chart(filtered_df)
            st.plotly_chart(fig_dept, use_container_width=True)
            
            # Executive summary table
            st.subheader("📊 Executive Summary by Department")
            
            st.dataframe(
                executive_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Department_Growth_Rate": st.column_config.TextColumn(
                        "Growth Rate",
                        help="Department growth rate based on business performance"
                    ),
                    "Total_Hires_Required": st.column_config.NumberColumn(
                        "Total Hires",
                        help="Replacement + Growth hires needed",
                        format="%.0f"
                    )
                }
            )
            
            # Department details
            if selected_dept != 'All Departments':
                st.subheader(f"📋 {selected_dept} - Detailed Hiring Plan")
                dept_data = filtered_df[filtered_df['Department'] == selected_dept]
                
                for idx, item in dept_data.iterrows():
                    display_hiring_item(item, show_employee_details=True)
    
    with tab4:
        st.header("👥 Employee Details & Replacement Planning")
        
        if at_risk_employees is not None:
            st.subheader("🚨 At-Risk Employees (Predicted Departures)")
            
            # Risk level filter for this tab
            risk_levels = ['All Risk Levels'] + sorted(at_risk_employees['Risk_Category'].unique().tolist())
            risk_filter = st.selectbox("Filter by Risk Level:", risk_levels, key="employee_risk_filter")
            
            filtered_at_risk = at_risk_employees.copy()
            if risk_filter != 'All Risk Levels':
                filtered_at_risk = filtered_at_risk[filtered_at_risk['Risk_Category'] == risk_filter]
            
            # Display at-risk employees
            st.dataframe(
                filtered_at_risk,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Attrition_Probability": st.column_config.NumberColumn(
                        "Attrition Probability",
                        help="ML model's confidence in departure prediction",
                        format="%.1%"
                    ),
                    "Estimated_Departure_Date": st.column_config.DateColumn(
                        "Estimated Departure",
                        help="Predicted departure date from ML model"
                    )
                }
            )
            
            # Employee search
            st.subheader("🔍 Individual Employee Lookup")
            
            employee_ids = ['Select Employee...'] + sorted(filtered_at_risk['Employee_ID'].tolist())
            selected_employee = st.selectbox("Search for specific employee:", employee_ids)
            
            if selected_employee != 'Select Employee...':
                employee_data = filtered_at_risk[filtered_at_risk['Employee_ID'] == selected_employee]
                
                if len(employee_data) > 0:
                    emp = employee_data.iloc[0]
                    
                    # Find corresponding hiring plan
                    hiring_plan = hiring_timeline[hiring_timeline['Employee_ID'] == selected_employee]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**👤 Employee Information**")
                        st.write(f"**Name:** {emp['Name']}")
                        st.write(f"**Department:** {emp['Department']}")
                        st.write(f"**Risk Level:** {emp['Risk_Category']}")
                        st.write(f"**Departure Probability:** {emp['Attrition_Probability']:.1%}")
                        if pd.notna(emp['Estimated_Departure_Date']):
                            st.write(f"**Estimated Departure:** {pd.to_datetime(emp['Estimated_Departure_Date']).strftime('%Y-%m-%d')}")
                    
                    with col2:
                        if len(hiring_plan) > 0:
                            plan = hiring_plan.iloc[0]
                            st.markdown("**📋 Replacement Plan**")
                            st.write(f"**Priority:** {plan['Priority']}")
                            st.write(f"**Start Hiring:** {plan['Start_Hiring_Date'].strftime('%Y-%m-%d')}")
                            st.write(f"**Action Status:** {plan['Action_Status']}")
                            st.write(f"**Lead Time:** {plan['Lead_Time_Days']} days")
                            
                            # Urgency indicator
                            if plan['Days_From_Today'] <= 0:
                                st.error("🚨 START HIRING IMMEDIATELY!")
                            elif plan['Days_From_Today'] <= 7:
                                st.warning(f"⚠️ Start hiring in {plan['Days_From_Today']} days")
                            else:
                                st.info(f"📅 Start hiring in {plan['Days_From_Today']} days")
                        else:
                            st.write("No hiring plan found for this employee.")
    
    with tab5:
        st.header("📈 Hiring Analytics & Insights")
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        insights = []
        
        # Calculate insights
        total_hiring = len(hiring_timeline)
        replacement_pct = (len(hiring_timeline[hiring_timeline['Hiring_Type'] == 'Replacement']) / total_hiring * 100) if total_hiring > 0 else 0
        growth_pct = (len(hiring_timeline[hiring_timeline['Hiring_Type'] == 'Growth']) / total_hiring * 100) if total_hiring > 0 else 0
        critical_pct = (len(hiring_timeline[hiring_timeline['Priority'] == 'CRITICAL']) / total_hiring * 100) if total_hiring > 0 else 0
        
        insights.append(f"📊 **{total_hiring} total positions** need to be filled")
        insights.append(f"🔄 **{replacement_pct:.1f}%** are replacement hires (attrition-driven)")
        insights.append(f"📈 **{growth_pct:.1f}%** are growth hires (business expansion)")
        insights.append(f"🚨 **{critical_pct:.1f}%** are critical priority positions")
        
        # Department with most hiring needs
        dept_counts = hiring_timeline['Department'].value_counts()
        if len(dept_counts) > 0:
            insights.append(f"🏢 **{dept_counts.index[0]}** has the highest hiring need ({dept_counts.iloc[0]} positions)")
        
        # Timeline insights
        immediate_pct = (len(hiring_timeline[hiring_timeline['Days_From_Today'] <= 0]) / total_hiring * 100) if total_hiring > 0 else 0
        if immediate_pct > 0:
            insights.append(f"⚡ **{immediate_pct:.1f}%** of positions need immediate action")
        
        for insight in insights:
            st.markdown(insight)
        
        # Performance tracking
        st.subheader("📊 Hiring Performance Tracking")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Lead time analysis
            lead_time_stats = hiring_timeline.groupby('Department')['Lead_Time_Days'].agg(['mean', 'min', 'max']).round(0)
            lead_time_stats.columns = ['Avg Lead Time', 'Min Lead Time', 'Max Lead Time']
            
            st.markdown("**Lead Time Analysis by Department**")
            st.dataframe(lead_time_stats, use_container_width=True)
        
        with col2:
            # Hiring volume by month
            hiring_timeline['Start_Month'] = hiring_timeline['Start_Hiring_Date'].dt.to_period('M')
            monthly_hiring = hiring_timeline.groupby(['Start_Month', 'Hiring_Type']).size().reset_index(name='Count')
            monthly_hiring['Start_Month'] = monthly_hiring['Start_Month'].astype(str)
            
            fig_monthly = px.bar(
                monthly_hiring.head(12),  # Show next 12 months
                x='Start_Month',
                y='Count',
                color='Hiring_Type',
                title="Hiring Volume by Month",
                color_discrete_map={
                    'Replacement': '#d32f2f',
                    'Growth': '#1976d2'
                }
            )
            
            fig_monthly.update_layout(
                title={'x': 0.5, 'xanchor': 'center'},
                height=300
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Export options
        st.subheader("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Full hiring timeline
            csv_all = hiring_timeline.to_csv(index=False)
            st.download_button(
                label="📄 Download Complete Hiring Plan",
                data=csv_all,
                file_name=f'complete_hiring_plan_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        
        with col2:
            # Immediate actions only
            immediate_actions = hiring_timeline[hiring_timeline['Days_From_Today'] <= 7]
            csv_immediate = immediate_actions.to_csv(index=False)
            st.download_button(
                label="🚨 Download Immediate Actions",
                data=csv_immediate,
                file_name=f'immediate_actions_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        
        with col3:
            # Executive summary
            if executive_summary is not None:
                csv_summary = executive_summary.to_csv(index=False)
                st.download_button(
                    label="📊 Download Executive Summary",
                    data=csv_summary,
                    file_name=f'executive_summary_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()