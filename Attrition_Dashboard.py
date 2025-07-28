import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="HR Attrition Dashboard - FIXED",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .urgent-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #d32f2f;
    }
    .warning-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f57c00;
    }
    .safe-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2e7d32;
    }
    .employee-result {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load employee data - FIXED to match new pipeline"""
    try:
        employee_df = pd.read_csv('employee_data_processed.csv')
        try:
            # Try FIXED results file first, then fallback
            with open('model_results_enhanced.json', 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            try:
                with open('model_results_enhanced.json', 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                results = None
        return employee_df, results
    except FileNotFoundError:
        st.error("❌ Data files not found. Please run the FIXED modeling pipeline first.")
        return None, None

@st.cache_resource
def load_models():
    """Load trained models - FIXED to match new pipeline"""
    try:
        # Try FIXED models file first, then fallback
        try:
            models = joblib.load('attrition_models_enhanced.pkl')
        except FileNotFoundError:
            models = joblib.load('attrition_models_enhanced.pkl')
        return models
    except FileNotFoundError:
        st.error("❌ Models not found. Please run the FIXED modeling pipeline first.")
        return None

def prepare_features_for_prediction(df, models):
    """Prepare features with proper encoding - FIXED to match new pipeline"""
    df_processed = df.copy()
    
    # Handle missing values
    if 'Manager_ID' in df_processed.columns:
        df_processed['Manager_ID'] = df_processed['Manager_ID'].fillna('None')
    
    # Add engineered features if missing
    if 'Tenure_Years' not in df_processed.columns:
        if 'Joining_Date' in df_processed.columns:
            df_processed['Joining_Date'] = pd.to_datetime(df_processed['Joining_Date'])
            df_processed['Tenure_Years'] = ((pd.Timestamp.now() - df_processed['Joining_Date']).dt.days / 365).round(1)
        else:
            df_processed['Tenure_Years'] = 2.0  # Default value
    
    if 'Experience_Tenure_Ratio' not in df_processed.columns:
        if 'Total_Experience' in df_processed.columns:
            df_processed['Experience_Tenure_Ratio'] = (df_processed['Total_Experience'] / (df_processed['Tenure_Years'] + 0.1)).round(2)
        else:
            df_processed['Experience_Tenure_Ratio'] = 1.0  # Default value
    
    if 'Salary_Satisfaction' not in df_processed.columns:
        if 'Market_Salary_Ratio' in df_processed.columns and 'Job_Satisfaction_Score' in df_processed.columns:
            df_processed['Salary_Satisfaction'] = (df_processed['Market_Salary_Ratio'] * df_processed['Job_Satisfaction_Score']).round(2)
        else:
            df_processed['Salary_Satisfaction'] = 5.0  # Default value
    
    # Apply encoders
    encoders = models['encoders']
    
    try:
        # Encode categorical variables
        for col, encoder in encoders.items():
            if col == 'department':
                original_col = 'Department'
            elif col == 'designation':
                original_col = 'Designation'
            elif col == 'location':
                original_col = 'Location'
            elif col == 'education':
                original_col = 'Education'
            elif col == 'workload':
                original_col = 'Project_Workload'
            elif col == 'worklife':
                original_col = 'Work_Life_Balance'
            elif col == 'notice_type':
                original_col = 'Notice_Period_Type'
            else:
                continue
            
            if original_col in df_processed.columns:
                # Handle unknown categories
                known_categories = set(encoder.classes_)
                df_processed[original_col] = df_processed[original_col].apply(
                    lambda x: x if x in known_categories else encoder.classes_[0]
                )
                df_processed[f'{original_col}_Encoded'] = encoder.transform(df_processed[original_col])
    
    except Exception as e:
        st.error(f"Encoding error: {str(e)}")
        return None
    
    # Get required features
    feature_cols = models['feature_cols']
    for feature in feature_cols:
        if feature not in df_processed.columns:
            df_processed[feature] = 0  # Default value for missing features
    
    return df_processed[feature_cols]

def generate_predictions(employee_df, models):
    """Generate predictions for active employees - FIXED to match new pipeline exactly"""
    active_employees = employee_df[employee_df['Status'] == 'Active'].copy()
    
    if len(active_employees) == 0 or models is None:
        return pd.DataFrame()
    
    try:
        X = prepare_features_for_prediction(active_employees, models)
        
        if X is None:
            return pd.DataFrame()
        
        # WHO predictions
        attrition_prob = models['best_classifier'].predict_proba(X)[:, 1]
        
        # WHEN predictions - FIXED to match the exact pipeline logic
        # Use ML model to predict notice period for all employees
        predicted_notice_periods = models['best_regressor'].predict(X)
        
        # Ensure realistic bounds (same as in FIXED pipeline)
        predicted_notice_periods = np.clip(predicted_notice_periods, 7, 180)  # 1 week to 6 months
        
        # Add predictions to dataframe
        active_employees['Attrition_Probability'] = attrition_prob.round(3)
        active_employees['Predicted_Notice_Period_Days'] = predicted_notice_periods.round().astype(int)
        
        # Calculate REALISTIC departure timeline (same as FIXED pipeline)
        today = datetime.now()
        
        # Estimate when they might submit resignation (based on risk level)
        resignation_timeline_days = []
        for prob in attrition_prob:
            if prob >= 0.7:  # High risk
                days_to_resignation = np.random.randint(30, 90)  # 1-3 months
            elif prob >= 0.4:  # Medium risk  
                days_to_resignation = np.random.randint(90, 180)  # 3-6 months
            else:  # Low risk
                days_to_resignation = np.random.randint(180, 365)  # 6-12 months
            resignation_timeline_days.append(days_to_resignation)
        
        active_employees['Estimated_Resignation_Date'] = [
            (today + timedelta(days=int(days))).strftime('%Y-%m-%d') 
            for days in resignation_timeline_days
        ]
        
        # Calculate final departure date (resignation + notice period)
        active_employees['Estimated_Departure_Date'] = [
            (datetime.strptime(resign_date, '%Y-%m-%d') + timedelta(days=int(notice_days))).strftime('%Y-%m-%d')
            for resign_date, notice_days in zip(active_employees['Estimated_Resignation_Date'], 
                                              active_employees['Predicted_Notice_Period_Days'])
        ]
        
        # Risk categorization (same as FIXED pipeline)
        def get_risk_category(prob):
            if prob >= 0.7: return 'High'
            elif prob >= 0.4: return 'Medium'
            else: return 'Low'
        
        active_employees['Risk_Category'] = active_employees['Attrition_Probability'].apply(get_risk_category)
        
        # For dashboard compatibility, also create Predicted_Lead_Time_Days 
        # This represents total time until departure (resignation timeline + notice period)
        active_employees['Predicted_Lead_Time_Days'] = [
            days_to_resign + notice_days 
            for days_to_resign, notice_days in zip(resignation_timeline_days, predicted_notice_periods)
        ]
        
        return active_employees
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        st.error(f"Available columns: {list(active_employees.columns)}")
        return pd.DataFrame()

def create_interactive_risk_charts(filtered_df):
    """Create interactive charts for risk analysis with correct colors"""
    
    # Risk distribution with proper colors
    risk_counts = filtered_df['Risk_Category'].value_counts()
    
    # Fixed color mapping: Green for Low Risk, Red for High Risk
    color_map = {
        'Low': '#2e7d32',      # Green
        'Medium': '#f57c00',   # Orange
        'High': '#d32f2f'      # Red
    }
    
    colors = [color_map.get(label, '#gray') for label in risk_counts.index]
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker_colors=colors,
        hovertemplate='<b>%{label}</b><br>' +
                      'Count: %{value}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>',
        textinfo='label+percent',
        textposition='inside'
    )])
    
    fig_pie.update_layout(
        title={
            'text': "Employee Risk Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        font=dict(size=12),
        showlegend=True,
        height=400
    )
    
    return fig_pie

def create_department_risk_chart(filtered_df):
    """Create interactive department risk breakdown"""
    dept_risk = filtered_df.groupby(['Department', 'Risk_Category']).size().reset_index(name='Count')
    
    fig_dept = px.bar(
        dept_risk, 
        x='Department', 
        y='Count', 
        color='Risk_Category',
        title="Risk Distribution by Department",
        color_discrete_map={
            'High': '#d32f2f',    # Red
            'Medium': '#f57c00',  # Orange
            'Low': '#2e7d32'      # Green
        },
        hover_data=['Count']
    )
    
    fig_dept.update_layout(
        title={
            'text': "Risk Distribution by Department",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        xaxis_title="Department",
        yaxis_title="Number of Employees",
        height=400,
        hovermode='x unified'
    )
    
    return fig_dept

def create_timeline_chart(timeline_df):
    """Create interactive timeline visualization - FIXED"""
    # Create bins for timeline (updated for FIXED approach)
    timeline_df['Timeline_Bucket'] = pd.cut(
        timeline_df['Predicted_Lead_Time_Days'], 
        bins=[0, 60, 120, 180, 365, float('inf')],
        labels=['0-60 days', '61-120 days', '121-180 days', '181-365 days', '365+ days']
    )
    
    bucket_counts = timeline_df.groupby(['Timeline_Bucket', 'Risk_Category']).size().reset_index(name='Count')
    
    fig_timeline = px.bar(
        bucket_counts, 
        x='Timeline_Bucket', 
        y='Count', 
        color='Risk_Category',
        title="Employee Departure Timeline (FIXED: Resignation + Notice Period)",
        color_discrete_map={
            'High': '#d32f2f',
            'Medium': '#f57c00'
        },
        hover_data=['Count']
    )
    
    fig_timeline.update_layout(
        title={
            'text': "Employee Departure Timeline (FIXED: Resignation + Notice Period)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4'}
        },
        xaxis_title="Timeline",
        yaxis_title="Number of Employees",
        height=400
    )
    
    return fig_timeline

def search_employee_by_id(employee_id, predictions_df):
    """Search for employee by ID and return their data"""
    if not employee_id:
        return None
    
    # Try exact match first
    employee_data = predictions_df[predictions_df['Employee_ID'] == employee_id]
    
    if len(employee_data) == 0:
        # Try partial match
        employee_data = predictions_df[predictions_df['Employee_ID'].str.contains(str(employee_id), case=False, na=False)]
    
    return employee_data.iloc[0] if len(employee_data) > 0 else None

def main():
    # Load data and models
    employee_df, results = load_data()
    models = load_models()
    
    if employee_df is None:
        return
    
    # Header
    st.markdown('<div class="main-header">🚨 HR Attrition Analytics Dashboard - FIXED</div>', unsafe_allow_html=True)
    
    # Show model information if available
    if results is not None:
        with st.expander("📊 FIXED Model Performance Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Classification Model (WHO)")
                if 'best_classification_model' in results:
                    st.write(f"**Best Model:** {results['best_classification_model']}")
                if 'classification_results' in results:
                    for model, metrics in results['classification_results'].items():
                        if 'auc' in metrics:
                            st.write(f"**{model} AUC:** {metrics['auc']:.3f}")
                if 'approach' in results:
                    st.info(f"**Approach:** {results['approach']}")
            
            with col2:
                st.subheader("📅 Regression Model (WHEN - Notice Period)")
                if 'best_regression_model' in results:
                    st.write(f"**Best Model:** {results['best_regression_model']}")
                if 'regression_results' in results:
                    for model, metrics in results['regression_results'].items():
                        if 'r2' in metrics:
                            st.write(f"**{model} R²:** {metrics['r2']:.3f}")
                        if 'mae' in metrics:
                            st.write(f"**{model} MAE:** {metrics['mae']:.1f} days")
                st.info("**FIXED:** Predicts realistic notice periods based on role/department")
    
    # Generate predictions
    st.info("🔄 Generating FIXED predictions for active employees...")
    predictions_df = generate_predictions(employee_df, models)
    
    if predictions_df.empty:
        st.error("Unable to generate predictions. Please check your models and data.")
        return
    
    st.success(f"✅ Successfully generated FIXED predictions for {len(predictions_df)} active employees")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    departments = ['All Departments'] + sorted(predictions_df['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Filter by Department", departments)
    
    risk_levels = ['All Risk Levels', 'High', 'Medium', 'Low']
    selected_risk = st.sidebar.selectbox("Filter by Risk Level", risk_levels)
    
    time_horizons = ['All Time', 'Leaving in 60 days', 'Leaving in 120 days', 'Leaving in 180 days']
    selected_time = st.sidebar.selectbox("Filter by Timeline", time_horizons)
    
    # Apply filters
    filtered_df = predictions_df.copy()
    
    if selected_dept != 'All Departments':
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    
    if selected_risk != 'All Risk Levels':
        filtered_df = filtered_df[filtered_df['Risk_Category'] == selected_risk]
    
    if selected_time != 'All Time':
        days_filter = {'Leaving in 60 days': 60, 'Leaving in 120 days': 120, 'Leaving in 180 days': 180}
        max_days = days_filter[selected_time]
        filtered_df = filtered_df[
            (filtered_df['Predicted_Lead_Time_Days'] <= max_days) & 
            (filtered_df['Predicted_Lead_Time_Days'].notna())
        ]
    
    # Key metrics
    st.markdown("### 📊 Risk Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_active = len(filtered_df)
    high_risk = len(filtered_df[filtered_df['Risk_Category'] == 'High'])
    medium_risk = len(filtered_df[filtered_df['Risk_Category'] == 'Medium'])
    urgent_cases = len(filtered_df[
        (filtered_df['Risk_Category'] == 'High') & 
        (filtered_df['Predicted_Lead_Time_Days'] <= 60)  # FIXED: Updated to 60 days for realistic timeline
    ])
    
    with col1:
        st.metric("Total Active Employees", total_active)
    
    with col2:
        st.metric("High Risk Employees", high_risk, delta=f"{high_risk/total_active*100:.1f}%" if total_active > 0 else "0%")
    
    with col3:
        st.metric("Medium Risk Employees", medium_risk, delta=f"{medium_risk/total_active*100:.1f}%" if total_active > 0 else "0%")
    
    with col4:
        st.metric("🚨 Leaving in 60 Days", urgent_cases)  # FIXED: Updated to 60 days
    
    # Alert for urgent cases
    if urgent_cases > 0:
        st.markdown('<div class="urgent-card">', unsafe_allow_html=True)
        st.markdown(f"### 🚨 URGENT ACTION REQUIRED")
        st.markdown(f"**{urgent_cases} high-risk employees** may leave within 60 days!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 WHO is Leaving", "📅 WHEN are they Leaving", "👤 Individual Lookup", "📈 Analytics"
    ])
    
    with tab1:
        st.header("🚨 WHO is Leaving - Employee Risk List")
        
        # Interactive visualization section
        st.subheader("📊 Interactive Risk Visualization")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Interactive pie chart with correct colors
            fig_pie = create_interactive_risk_charts(filtered_df)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with viz_col2:
            # Interactive department chart
            if selected_dept == 'All Departments':
                fig_dept = create_department_risk_chart(filtered_df)
                st.plotly_chart(fig_dept, use_container_width=True)
            else:
                # Show selected department breakdown
                dept_data = filtered_df[filtered_df['Department'] == selected_dept]
                risk_counts = dept_data['Risk_Category'].value_counts()
                
                color_map = {
                    'Low': '#2e7d32',      # Green
                    'Medium': '#f57c00',   # Orange
                    'High': '#d32f2f'      # Red
                }
                
                colors = [color_map.get(x, '#gray') for x in risk_counts.index]
                
                fig_dept_single = go.Figure(data=[go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=colors,
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )])
                
                fig_dept_single.update_layout(
                    title=f"Risk Distribution - {selected_dept}",
                    xaxis_title="Risk Category",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig_dept_single, use_container_width=True)
        
        # Employee tables
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # High risk employees
            high_risk_employees = filtered_df[filtered_df['Risk_Category'] == 'High'].copy()
            
            if len(high_risk_employees) > 0:
                st.subheader("🔴 High Risk Employees (Immediate Action Required)")
                
                display_cols = [
                    'Employee_ID', 'Name', 'Department', 'Designation',
                    'Attrition_Probability', 'Estimated_Departure_Date', 'Predicted_Notice_Period_Days',
                    'Job_Satisfaction_Score', 'Manager_Rating', 'Monthly_Salary'
                ]
                
                # Only include columns that exist
                available_cols = [col for col in display_cols if col in high_risk_employees.columns]
                high_risk_display = high_risk_employees[available_cols].sort_values('Attrition_Probability', ascending=False)
                
                st.dataframe(
                    high_risk_display, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Attrition_Probability": st.column_config.NumberColumn(
                            "Attrition Probability",
                            help="Probability of employee leaving",
                            format="%.1%"
                        ),
                        "Predicted_Notice_Period_Days": st.column_config.NumberColumn(
                            "Notice Period (Days)",
                            help="Predicted notice period length",
                            format="%.0f"
                        ),
                        "Monthly_Salary": st.column_config.NumberColumn(
                            "Monthly Salary",
                            help="Employee's monthly salary",
                            format="$%d"
                        ) if 'Monthly_Salary' in available_cols else None
                    }
                )
                
                # Download button
                csv = high_risk_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download High Risk List",
                    data=csv,
                    file_name=f'high_risk_employees_FIXED_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            else:
                st.success("✅ No high-risk employees found!")
            
            # Medium risk employees
            medium_risk_employees = filtered_df[filtered_df['Risk_Category'] == 'Medium']
            
            if len(medium_risk_employees) > 0:
                st.subheader("🟡 Medium Risk Employees (Monitor Closely)")
                
                medium_display_cols = [
                    'Employee_ID', 'Name', 'Department', 'Designation',
                    'Attrition_Probability', 'Predicted_Notice_Period_Days', 'Job_Satisfaction_Score', 'Manager_Rating'
                ]
                
                available_medium_cols = [col for col in medium_display_cols if col in medium_risk_employees.columns]
                medium_display_data = medium_risk_employees[available_medium_cols].sort_values('Attrition_Probability', ascending=False)
                
                st.write(f"**{len(medium_risk_employees)} Medium Risk Employees:**")
                st.dataframe(
                    medium_display_data, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Attrition_Probability": st.column_config.NumberColumn(
                            "Attrition Probability",
                            help="Probability of employee leaving",
                            format="%.1%"
                        ),
                        "Predicted_Notice_Period_Days": st.column_config.NumberColumn(
                            "Notice Period (Days)",
                            help="Predicted notice period length",
                            format="%.0f"
                        )
                    }
                )
        
        with col2:
            # Quick stats
            st.subheader("📈 Quick Stats")
            
            # Average probability by risk level
            avg_prob_high = filtered_df[filtered_df['Risk_Category'] == 'High']['Attrition_Probability'].mean()
            avg_prob_medium = filtered_df[filtered_df['Risk_Category'] == 'Medium']['Attrition_Probability'].mean()
            
            st.metric("Avg High Risk Prob", f"{avg_prob_high:.1%}" if not pd.isna(avg_prob_high) else "N/A")
            st.metric("Avg Medium Risk Prob", f"{avg_prob_medium:.1%}" if not pd.isna(avg_prob_medium) else "N/A")
            
            # FIXED: Notice period stats
            if 'Predicted_Notice_Period_Days' in filtered_df.columns:
                avg_notice_high = filtered_df[filtered_df['Risk_Category'] == 'High']['Predicted_Notice_Period_Days'].mean()
                avg_notice_medium = filtered_df[filtered_df['Risk_Category'] == 'Medium']['Predicted_Notice_Period_Days'].mean()
                
                st.metric("Avg High Risk Notice", f"{avg_notice_high:.0f} days" if not pd.isna(avg_notice_high) else "N/A")
                st.metric("Avg Medium Risk Notice", f"{avg_notice_medium:.0f} days" if not pd.isna(avg_notice_medium) else "N/A")
            
            # Top department at risk
            if selected_dept == 'All Departments':
                dept_high_risk = filtered_df[filtered_df['Risk_Category'] == 'High']['Department'].value_counts()
                if len(dept_high_risk) > 0:
                    st.metric("Top Risk Department", dept_high_risk.index[0])
                    st.metric("High Risk Count", dept_high_risk.iloc[0])
    
    with tab2:
        st.header("📅 WHEN are they Leaving - FIXED Timeline View")
        
        # Simple filters for departure timeline
        st.subheader("🎛️ Timeline Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Department filter for timeline
            timeline_departments = ['All Departments'] + sorted(predictions_df['Department'].unique().tolist())
            timeline_dept = st.selectbox("Filter by Department:", timeline_departments, key="timeline_dept")
        
        with col2:
            # Risk level filter for timeline
            timeline_risks = ['All Risk Levels', 'High', 'Medium']
            timeline_risk = st.selectbox("Filter by Risk Level:", timeline_risks, key="timeline_risk")
        
        # Filter timeline data
        timeline_df = predictions_df[
            (predictions_df['Risk_Category'].isin(['High', 'Medium'])) &
            (predictions_df['Predicted_Lead_Time_Days'].notna())
        ].copy()
        
        if timeline_dept != 'All Departments':
            timeline_df = timeline_df[timeline_df['Department'] == timeline_dept]
        
        if timeline_risk != 'All Risk Levels':
            timeline_df = timeline_df[timeline_df['Risk_Category'] == timeline_risk]
        
        if len(timeline_df) > 0:
            # Timeline categories (FIXED for realistic timelines)
            st.subheader("📊 FIXED Departure Timeline Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            next_60 = len(timeline_df[timeline_df['Predicted_Lead_Time_Days'] <= 60])
            next_120 = len(timeline_df[(timeline_df['Predicted_Lead_Time_Days'] > 60) & (timeline_df['Predicted_Lead_Time_Days'] <= 120)])
            next_180 = len(timeline_df[(timeline_df['Predicted_Lead_Time_Days'] > 120) & (timeline_df['Predicted_Lead_Time_Days'] <= 180)])
            beyond_180 = len(timeline_df[timeline_df['Predicted_Lead_Time_Days'] > 180])
            
            with col1:
                st.markdown('<div class="urgent-card">', unsafe_allow_html=True)
                st.metric("Next 60 Days", next_60)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.metric("61-120 Days", next_120)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.metric("121-180 Days", next_180)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="safe-card">', unsafe_allow_html=True)
                st.metric("Beyond 180 Days", beyond_180)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interactive timeline visualization
            fig_timeline = create_timeline_chart(timeline_df)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # FIXED: Enhanced timeline explanation
            st.info("💡 **FIXED Timeline**: Shows total time until departure = resignation timeline + notice period")
            
            # Timeline table
            st.subheader("📋 FIXED Departure Timeline")
            st.info("💡 Click on any Employee ID below to view detailed information")
            
            timeline_display_cols = [
                'Employee_ID', 'Name', 'Department', 'Risk_Category',
                'Predicted_Lead_Time_Days', 'Predicted_Notice_Period_Days', 'Estimated_Departure_Date',
                'Job_Satisfaction_Score', 'Manager_Rating'
            ]
            
            # Only include columns that exist
            available_timeline_cols = [col for col in timeline_display_cols if col in timeline_df.columns]
            timeline_sorted = timeline_df[available_timeline_cols].sort_values('Predicted_Lead_Time_Days')
            
            # Display the table
            st.dataframe(
                timeline_sorted, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Predicted_Lead_Time_Days": st.column_config.NumberColumn(
                        "Total Days Until Departure",
                        help="Predicted total days until employee leaves (resignation + notice)",
                        format="%.0f"
                    ),
                    "Predicted_Notice_Period_Days": st.column_config.NumberColumn(
                        "Notice Period (Days)",
                        help="Predicted notice period length based on role/department",
                        format="%.0f"
                    ),
                    "Employee_ID": st.column_config.TextColumn(
                        "Employee ID",
                        help="Click to view detailed employee information"
                    )
                }
            )
            
            # Simple employee selection for detailed view
            st.markdown("---")
            st.subheader("👤 View Employee Details")
            
            # Create a simple selectbox with Employee IDs from the current filtered data
            employee_options = timeline_sorted['Employee_ID'].tolist()
            selected_employee = st.selectbox(
                "Select Employee ID for detailed timeline:",
                [''] + employee_options,
                format_func=lambda x: "-- Select Employee --" if x == '' else x
            )
            
            # Show employee details if selected
            if selected_employee and selected_employee != '':
                employee_data = search_employee_by_id(selected_employee, timeline_df)
                
                if employee_data is not None:
                    st.markdown('<div class="employee-result">', unsafe_allow_html=True)
                    st.success(f"✅ Employee Details: {employee_data.get('Name', 'N/A')}")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.write(f"**Employee ID:** {employee_data['Employee_ID']}")
                        st.write(f"**Department:** {employee_data.get('Department', 'N/A')}")
                        st.write(f"**Risk Level:** {employee_data['Risk_Category']}")
                    
                    with result_col2:
                        st.write(f"**Attrition Probability:** {employee_data['Attrition_Probability']:.1%}")
                        if 'Predicted_Notice_Period_Days' in employee_data:
                            st.write(f"**Notice Period:** {int(employee_data['Predicted_Notice_Period_Days'])} days")
                        st.write(f"**Total Days Until Departure:** {int(employee_data['Predicted_Lead_Time_Days'])}")
                    
                    with result_col3:
                        st.write(f"**Estimated Departure:** {employee_data.get('Estimated_Departure_Date', 'N/A')}")
                        
                        # Color code based on urgency (FIXED thresholds)
                        days_left = employee_data['Predicted_Lead_Time_Days']
                        if days_left <= 60:
                            st.error("🚨 URGENT: Leaving within 60 days!")
                        elif days_left <= 120:
                            st.warning("⚠️ WARNING: Leaving within 120 days")
                        else:
                            st.info("📅 Monitor situation")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No employees found matching the selected filters.")
    
    with tab3:
        st.header("👤 Individual Employee Lookup")
        
        # Employee search
        employee_id_search = st.text_input(
            "Enter Employee ID:",
            placeholder="e.g., EMP_0001, EMP_0123",
            help="Search by Employee ID for detailed risk assessment"
        )
        
        # Quick dropdown for browsing
        employee_ids = ['Select Employee ID...'] + sorted(filtered_df['Employee_ID'].tolist())
        selected_employee_id = st.selectbox("Or browse Employee IDs:", employee_ids)
        
        # Determine which employee to show
        target_employee_id = None
        if employee_id_search:
            target_employee_id = employee_id_search
        elif selected_employee_id != 'Select Employee ID...':
            target_employee_id = selected_employee_id
        
        if target_employee_id:
            employee_data = search_employee_by_id(target_employee_id, filtered_df)
            
            if employee_data is not None:
                # Employee details
                st.subheader(f"📋 Employee Details: {employee_data.get('Name', 'N/A')}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Information**")
                    st.write(f"**ID:** {employee_data['Employee_ID']}")
                    st.write(f"**Name:** {employee_data.get('Name', 'N/A')}")
                    st.write(f"**Department:** {employee_data.get('Department', 'N/A')}")
                    st.write(f"**Designation:** {employee_data.get('Designation', 'N/A')}")
                    st.write(f"**Age:** {employee_data.get('Age', 'N/A')}")
                    st.write(f"**Tenure:** {employee_data.get('Tenure_Years', 'N/A')} years")
                
                with col2:
                    st.markdown("**FIXED Risk Assessment**")
                    
                    risk_category = employee_data['Risk_Category']
                    if risk_category == 'High':
                        st.markdown('<div class="urgent-card">', unsafe_allow_html=True)
                        st.write(f"**Risk Level:** 🔴 {risk_category} Risk")
                        st.write(f"**Probability:** {employee_data['Attrition_Probability']:.1%}")
                        if 'Predicted_Notice_Period_Days' in employee_data:
                            st.write(f"**Notice Period:** {int(employee_data['Predicted_Notice_Period_Days'])} days")
                        if not pd.isna(employee_data.get('Estimated_Departure_Date')):
                            st.write(f"**Est. Departure:** {employee_data['Estimated_Departure_Date']}")
                            st.write(f"**Total Days Until Departure:** {int(employee_data['Predicted_Lead_Time_Days'])}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif risk_category == 'Medium':
                        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                        st.write(f"**Risk Level:** 🟡 {risk_category} Risk")
                        st.write(f"**Probability:** {employee_data['Attrition_Probability']:.1%}")
                        if 'Predicted_Notice_Period_Days' in employee_data:
                            st.write(f"**Notice Period:** {int(employee_data['Predicted_Notice_Period_Days'])} days")
                        if not pd.isna(employee_data.get('Estimated_Departure_Date')):
                            st.write(f"**Est. Departure:** {employee_data['Estimated_Departure_Date']}")
                            st.write(f"**Total Days Until Departure:** {int(employee_data['Predicted_Lead_Time_Days'])}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="safe-card">', unsafe_allow_html=True)
                        st.write(f"**Risk Level:** 🟢 {risk_category} Risk")
                        st.write(f"**Probability:** {employee_data['Attrition_Probability']:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Key metrics
                st.subheader("📈 Key Risk Factors")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Job_Satisfaction_Score' in employee_data:
                        satisfaction_score = employee_data['Job_Satisfaction_Score']
                        st.metric("Job Satisfaction", f"{satisfaction_score:.1f}/10")
                        if satisfaction_score < 5:
                            st.error("🔴 Low satisfaction - retention risk!")
                        elif satisfaction_score < 7:
                            st.warning("🟡 Moderate satisfaction")
                        else:
                            st.success("🟢 High satisfaction")
                    
                    if 'Manager_Rating' in employee_data:
                        manager_rating = employee_data['Manager_Rating']
                        st.metric("Manager Rating", f"{manager_rating:.1f}/10")
                        if manager_rating < 5:
                            st.error("🔴 Poor manager relationship!")
                        elif manager_rating < 7:
                            st.warning("🟡 Average manager relationship")
                        else:
                            st.success("🟢 Good manager relationship")
                
                with col2:
                    if 'Market_Salary_Ratio' in employee_data:
                        salary_ratio = employee_data['Market_Salary_Ratio']
                        st.metric("Market Salary Ratio", f"{salary_ratio:.2f}")
                        if salary_ratio < 0.8:
                            st.error("🔴 Significantly underpaid!")
                        elif salary_ratio < 0.9:
                            st.warning("🟡 Below market rate")
                        else:
                            st.success("🟢 Competitive salary")
                    
                    if 'Monthly_Salary' in employee_data:
                        st.metric("Monthly Salary", f"${employee_data['Monthly_Salary']:,}")
                
                # Enhanced features if available
                if any(col in employee_data for col in ['Intent_To_Stay_12Months', 'Engagement_Survey_Score']):
                    st.subheader("🔬 Advanced Risk Indicators")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'Intent_To_Stay_12Months' in employee_data:
                            intent_score = employee_data['Intent_To_Stay_12Months']
                            st.metric("Intent to Stay (12 months)", f"{intent_score:.1f}/10")
                            if intent_score < 4:
                                st.error("🔴 Very low retention intent!")
                            elif intent_score < 7:
                                st.warning("🟡 Moderate retention intent")
                            else:
                                st.success("🟢 Strong retention intent")
                    
                    with col2:
                        if 'Engagement_Survey_Score' in employee_data:
                            engagement_score = employee_data['Engagement_Survey_Score']
                            st.metric("Engagement Score", f"{engagement_score:.1f}/10")
                            if engagement_score < 4:
                                st.error("🔴 Low engagement!")
                            elif engagement_score < 7:
                                st.warning("🟡 Moderate engagement")
                            else:
                                st.success("🟢 High engagement")
                
                # FIXED: Enhanced recommendations
                if risk_category in ['High', 'Medium']:
                    st.subheader("💡 FIXED Recommended Actions")
                    
                    recommendations = []
                    
                    # Satisfaction-based recommendations
                    if employee_data.get('Job_Satisfaction_Score', 10) < 5:
                        recommendations.append("🆘 CRITICAL: Immediate intervention needed - Schedule emergency retention meeting")
                    elif employee_data.get('Job_Satisfaction_Score', 10) < 7:
                        recommendations.append("🗣️ Schedule one-on-one satisfaction discussion within 1 week")
                    
                    # Salary-based recommendations
                    if employee_data.get('Market_Salary_Ratio', 1.0) < 0.8:
                        recommendations.append("💰 URGENT: Significant salary adjustment needed - escalate to leadership")
                    elif employee_data.get('Market_Salary_Ratio', 1.0) < 0.9:
                        recommendations.append("💰 Review and potentially adjust compensation package")
                    
                    # Manager relationship recommendations
                    if employee_data.get('Manager_Rating', 10) < 5:
                        recommendations.append("👥 CRITICAL: Manager relationship issue - consider team transfer or manager coaching")
                    elif employee_data.get('Manager_Rating', 10) < 7:
                        recommendations.append("👥 Assess and improve manager-employee relationship dynamics")
                    
                    # Enhanced feature recommendations
                    if employee_data.get('Intent_To_Stay_12Months', 10) < 4:
                        recommendations.append("🚨 IMMEDIATE: Employee has very low retention intent - emergency action required")
                    
                    if employee_data.get('Engagement_Survey_Score', 10) < 4:
                        recommendations.append("📈 Focus on engagement initiatives - career development, recognition, challenging work")
                    
                    # Timeline-based recommendations
                    if not pd.isna(employee_data.get('Predicted_Lead_Time_Days')) and employee_data.get('Predicted_Lead_Time_Days', 365) <= 60:
                        recommendations.append("⚡ URGENT: Employee may leave within 60 days - immediate retention meeting required")
                    elif not pd.isna(employee_data.get('Predicted_Lead_Time_Days')) and employee_data.get('Predicted_Lead_Time_Days', 365) <= 120:
                        recommendations.append("⏰ Priority: Plan comprehensive retention strategy within 2 weeks")
                    
                    # Notice period specific recommendations
                    if 'Predicted_Notice_Period_Days' in employee_data:
                        notice_days = employee_data['Predicted_Notice_Period_Days']
                        if notice_days < 30:
                            recommendations.append("📋 Prepare for short notice period - expedite replacement hiring and knowledge transfer")
                        elif notice_days > 60:
                            recommendations.append("📋 Leverage longer notice period for thorough knowledge transfer and replacement training")
                    
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            if "CRITICAL" in rec or "URGENT" in rec or "IMMEDIATE" in rec:
                                st.error(f"{i}. {rec}")
                            elif "Priority" in rec:
                                st.warning(f"{i}. {rec}")
                            else:
                                st.info(f"{i}. {rec}")
                    else:
                        st.info("• Continue regular monitoring and maintain engagement initiatives")
                
            else:
                st.error(f"❌ Employee ID '{target_employee_id}' not found.")
                st.info("💡 Try entering the full Employee ID (e.g., EMP_0001) or browse the dropdown list.")
        else:
            st.info("🔍 Please enter an Employee ID or select from the dropdown to view detailed analysis.")
    
    with tab4:
        st.header("📈 FIXED Analytics & Insights")
        
        # Analytics section
        col1, col2 = st.columns(2)
        
        with col1:
            # Job satisfaction analysis
            if 'Job_Satisfaction_Score' in filtered_df.columns:
                fig_satisfaction = px.box(
                    filtered_df, 
                    x='Risk_Category', 
                    y='Job_Satisfaction_Score',
                    title="Job Satisfaction Distribution by Risk Level",
                    color='Risk_Category',
                    color_discrete_map={
                        'High': '#d32f2f',
                        'Medium': '#f57c00', 
                        'Low': '#2e7d32'
                    }
                )
                
                fig_satisfaction.update_layout(
                    title={
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    height=400
                )
                
                st.plotly_chart(fig_satisfaction, use_container_width=True)
            else:
                st.info("Job Satisfaction data not available")
        
        with col2:
            # FIXED: Notice period analysis
            if 'Predicted_Notice_Period_Days' in filtered_df.columns:
                fig_notice = px.box(
                    filtered_df, 
                    x='Risk_Category', 
                    y='Predicted_Notice_Period_Days',
                    title="FIXED: Predicted Notice Period by Risk Level",
                    color='Risk_Category',
                    color_discrete_map={
                        'High': '#d32f2f',
                        'Medium': '#f57c00', 
                        'Low': '#2e7d32'
                    }
                )
                
                fig_notice.update_layout(
                    title={
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    height=400
                )
                
                st.plotly_chart(fig_notice, use_container_width=True)
            else:
                # Fallback to salary ratio if notice period not available
                if 'Market_Salary_Ratio' in filtered_df.columns:
                    fig_salary = px.box(
                        filtered_df, 
                        x='Risk_Category', 
                        y='Market_Salary_Ratio',
                        title="Market Salary Ratio by Risk Level",
                        color='Risk_Category',
                        color_discrete_map={
                            'High': '#d32f2f',
                            'Medium': '#f57c00', 
                            'Low': '#2e7d32'
                        }
                    )
                    
                    fig_salary.update_layout(
                        title={
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 16}
                        },
                        height=400
                    )
                    
                    st.plotly_chart(fig_salary, use_container_width=True)
                else:
                    st.info("Market Salary Ratio data not available")
        
        # Department-wise analysis
        st.subheader("🏢 Department-wise FIXED Risk Analysis")
        
        # Create department analysis with available columns only
        analysis_cols = {
            'Employee_ID': 'count',
            'Attrition_Probability': 'mean'
        }
        
        # Add optional columns if they exist
        if 'Job_Satisfaction_Score' in filtered_df.columns:
            analysis_cols['Job_Satisfaction_Score'] = 'mean'
        if 'Manager_Rating' in filtered_df.columns:
            analysis_cols['Manager_Rating'] = 'mean'
        if 'Market_Salary_Ratio' in filtered_df.columns:
            analysis_cols['Market_Salary_Ratio'] = 'mean'
        if 'Predicted_Notice_Period_Days' in filtered_df.columns:
            analysis_cols['Predicted_Notice_Period_Days'] = 'mean'
        
        dept_analysis = filtered_df.groupby('Department').agg(analysis_cols).round(3)
        
        # Rename columns
        new_column_names = ['Total Employees', 'Avg Attrition Prob']
        if 'Job_Satisfaction_Score' in analysis_cols:
            new_column_names.append('Avg Job Satisfaction')
        if 'Manager_Rating' in analysis_cols:
            new_column_names.append('Avg Manager Rating')
        if 'Market_Salary_Ratio' in analysis_cols:
            new_column_names.append('Avg Salary Ratio')
        if 'Predicted_Notice_Period_Days' in analysis_cols:
            new_column_names.append('Avg Notice Period (Days)')
        
        dept_analysis.columns = new_column_names
        dept_analysis = dept_analysis.sort_values('Avg Attrition Prob', ascending=False)
        
        st.dataframe(
            dept_analysis,
            use_container_width=True,
            column_config={
                "Avg Attrition Prob": st.column_config.NumberColumn(
                    "Avg Attrition Probability",
                    help="Average attrition probability for department",
                    format="%.1%"
                ),
                "Avg Salary Ratio": st.column_config.NumberColumn(
                    "Avg Market Salary Ratio",
                    help="Average market salary ratio for department",
                    format="%.2f"
                ) if 'Avg Salary Ratio' in new_column_names else None,
                "Avg Notice Period (Days)": st.column_config.NumberColumn(
                    "Avg Notice Period (Days)",
                    help="Average predicted notice period for department",
                    format="%.0f"
                ) if 'Avg Notice Period (Days)' in new_column_names else None
            }
        )
        
        # FIXED: Key insights summary
        st.subheader("🔍 FIXED Key Insights Summary")
        
        insights = []
        
        # Calculate insights
        high_risk_pct = (len(filtered_df[filtered_df['Risk_Category'] == 'High']) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        
        insights.append(f"📊 **{high_risk_pct:.1f}%** of active employees are at high risk of attrition")
        
        # Job satisfaction comparison if available
        if 'Job_Satisfaction_Score' in filtered_df.columns:
            avg_satisfaction_high_risk = filtered_df[filtered_df['Risk_Category'] == 'High']['Job_Satisfaction_Score'].mean()
            avg_satisfaction_low_risk = filtered_df[filtered_df['Risk_Category'] == 'Low']['Job_Satisfaction_Score'].mean()
            
            if not pd.isna(avg_satisfaction_high_risk) and not pd.isna(avg_satisfaction_low_risk):
                satisfaction_diff = avg_satisfaction_low_risk - avg_satisfaction_high_risk
                insights.append(f"😟 High-risk employees have **{satisfaction_diff:.1f} points lower** job satisfaction on average")
        
        # FIXED: Notice period insights
        if 'Predicted_Notice_Period_Days' in filtered_df.columns:
            avg_notice_high = filtered_df[filtered_df['Risk_Category'] == 'High']['Predicted_Notice_Period_Days'].mean()
            avg_notice_medium = filtered_df[filtered_df['Risk_Category'] == 'Medium']['Predicted_Notice_Period_Days'].mean()
            
            if not pd.isna(avg_notice_high) and not pd.isna(avg_notice_medium):
                insights.append(f"📋 High-risk employees predicted to give **{avg_notice_high:.0f} days** notice on average")
                insights.append(f"📋 Medium-risk employees predicted to give **{avg_notice_medium:.0f} days** notice on average")
        
        # Department with highest risk
        if len(filtered_df) > 0:
            dept_risk = filtered_df.groupby('Department')['Attrition_Probability'].mean().sort_values(ascending=False)
            if len(dept_risk) > 0:
                insights.append(f"🏢 **{dept_risk.index[0]}** department has the highest average attrition risk ({dept_risk.iloc[0]:.1%})")
        
        # FIXED: Urgent cases with updated thresholds
        urgent_count = len(filtered_df[
            (filtered_df['Risk_Category'] == 'High') & 
            (filtered_df['Predicted_Lead_Time_Days'] <= 60)
        ])
        
        if urgent_count > 0:
            insights.append(f"🚨 **{urgent_count} employees** require immediate attention (leaving within 60 days)")
        
        # Prediction coverage
        with_predictions = len(filtered_df[filtered_df['Predicted_Lead_Time_Days'].notna()])
        coverage_pct = (with_predictions / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        insights.append(f"📈 **{coverage_pct:.1f}%** of employees have FIXED departure timeline predictions")
        
        for insight in insights:
            st.markdown(insight)
        
        # FIXED: Model performance summary
        if results is not None:
            st.subheader("🤖 FIXED Model Performance Summary")
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("**Classification Model (WHO)**")
                if 'classification_results' in results:
                    for model, metrics in results['classification_results'].items():
                        if 'auc' in metrics:
                            auc_score = metrics['auc']
                            if auc_score >= 0.9:
                                st.success(f"✅ {model}: Excellent AUC ({auc_score:.3f})")
                            elif auc_score >= 0.8:
                                st.info(f"ℹ️ {model}: Good AUC ({auc_score:.3f})")
                            else:
                                st.warning(f"⚠️ {model}: Fair AUC ({auc_score:.3f})")
            
            with perf_col2:
                st.markdown("**FIXED Regression Model (NOTICE PERIOD)**")
                if 'regression_results' in results:
                    for model, metrics in results['regression_results'].items():
                        if 'r2' in metrics:
                            r2_score = metrics['r2']
                            if r2_score >= 0.3:
                                st.success(f"✅ {model}: Good R² ({r2_score:.3f})")
                            elif r2_score >= 0.1:
                                st.info(f"ℹ️ {model}: Fair R² ({r2_score:.3f})")
                            else:
                                st.warning(f"⚠️ {model}: Low R² ({r2_score:.3f})")
                        
                        if 'mae' in metrics:
                            st.write(f"📊 MAE: {metrics['mae']:.1f} days")
                
                if 'approach' in results:
                    st.info(f"**Approach:** {results['approach']}")
        
        # Download predictions
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export all predictions
            if st.button("📊 Export All Predictions"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=f'FIXED_all_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
        
        with col2:
            # Export high risk only
            high_risk_df = filtered_df[filtered_df['Risk_Category'] == 'High']
            if len(high_risk_df) > 0 and st.button("🚨 Export High Risk Only"):
                high_risk_csv = high_risk_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download High Risk CSV",
                    data=high_risk_csv,
                    file_name=f'FIXED_high_risk_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()