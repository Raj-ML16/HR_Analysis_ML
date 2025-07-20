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
    page_title="HR Attrition Dashboard",
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
    """Load employee data - Updated to match new pipeline"""
    try:
        employee_df = pd.read_csv('employee_data_processed.csv')
        try:
            # Try new results file first, then fallback to old
            with open('model_results_fixed.json', 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            try:
                with open('model_results.json', 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                results = None
        return employee_df, results
    except FileNotFoundError:
        st.error("❌ Data files not found. Please run the modeling pipeline first.")
        return None, None

@st.cache_resource
def load_models():
    """Load trained models - Updated to match new pipeline"""
    try:
        # Try new models file first, then fallback to old
        try:
            models = joblib.load('attrition_models_fixed.pkl')
        except FileNotFoundError:
            models = joblib.load('attrition_models.pkl')
        return models
    except FileNotFoundError:
        st.error("❌ Models not found. Please run the modeling pipeline first.")
        return None

def prepare_features_for_prediction(df, models):
    """Prepare features with proper encoding - Updated for new pipeline"""
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

def estimate_departure_timeline(attrition_prob, employee_id):
    """Estimate departure timeline - Same as in new pipeline"""
    # Use employee_id for unique seed to get different results per employee
    seed_value = hash(str(employee_id)) % 10000
    np.random.seed(seed_value)
    
    if attrition_prob >= 0.7:    # High risk
        base_days = 60  # 2 months average
        variation = np.random.randint(-20, 21)  # ±20 days variation
        return max(30, min(90, base_days + variation))
    elif attrition_prob >= 0.4:  # Medium risk  
        base_days = 135  # 4.5 months average
        variation = np.random.randint(-30, 31)  # ±30 days variation
        return max(90, min(180, base_days + variation))
    else:                        # Low risk
        base_days = 270  # 9 months average
        variation = np.random.randint(-60, 61)  # ±60 days variation
        return max(180, min(365, base_days + variation))

def generate_predictions(employee_df, models):
    """Generate predictions for active employees - Updated to match new pipeline logic"""
    active_employees = employee_df[employee_df['Status'] == 'Active'].copy()
    
    if len(active_employees) == 0 or models is None:
        return pd.DataFrame()
    
    try:
        X = prepare_features_for_prediction(active_employees, models)
        
        if X is None:
            return pd.DataFrame()
        
        # WHO predictions
        attrition_prob = models['best_classifier'].predict_proba(X)[:, 1]
        
        # WHEN predictions - Updated to match new pipeline logic
        lead_times = np.full(len(active_employees), np.nan)
        
        # Lower threshold: Medium+ risk employees (>= 0.4) get predictions
        medium_high_risk_mask = attrition_prob >= 0.4
        
        if np.any(medium_high_risk_mask):
            # Use ML model for high-risk employees (>= 0.5) - ROUND TO INTEGERS
            high_risk_indices = np.where((attrition_prob >= 0.5) & medium_high_risk_mask)[0]
            if len(high_risk_indices) > 0:
                ml_predictions = models['best_regressor'].predict(X.iloc[high_risk_indices])
                lead_times[high_risk_indices] = np.round(ml_predictions).astype(int)
            
            # Use risk-based estimation for medium-risk employees (0.4-0.5)
            medium_risk_indices = np.where((attrition_prob >= 0.4) & (attrition_prob < 0.5))[0]
            for idx in medium_risk_indices:
                employee_id = active_employees.iloc[idx]['Employee_ID']
                lead_times[idx] = estimate_departure_timeline(attrition_prob[idx], employee_id)
        
        # For low-risk employees who might still have some probability
        low_risk_indices = np.where((attrition_prob >= 0.2) & (attrition_prob < 0.4))[0]
        for idx in low_risk_indices:
            employee_id = active_employees.iloc[idx]['Employee_ID']
            lead_times[idx] = estimate_departure_timeline(attrition_prob[idx], employee_id)
        
        # Add predictions to dataframe
        active_employees['Attrition_Probability'] = attrition_prob.round(3)
        active_employees['Predicted_Lead_Time_Days'] = lead_times
        
        # Calculate departure date - For ALL employees with lead times
        today = datetime.now()
        active_employees['Estimated_Departure_Date'] = active_employees.apply(
            lambda row: (today + timedelta(days=int(row['Predicted_Lead_Time_Days']))).strftime('%Y-%m-%d') 
            if not pd.isna(row['Predicted_Lead_Time_Days']) else None, axis=1
        )
        
        # Risk categorization - Updated thresholds
        def get_risk_category(prob):
            if prob >= 0.7: return 'High'
            elif prob >= 0.4: return 'Medium'
            else: return 'Low'
        
        active_employees['Risk_Category'] = active_employees['Attrition_Probability'].apply(get_risk_category)
        
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
    """Create interactive timeline visualization"""
    # Create bins for timeline
    timeline_df['Timeline_Bucket'] = pd.cut(
        timeline_df['Predicted_Lead_Time_Days'], 
        bins=[0, 30, 60, 90, 180, float('inf')],
        labels=['0-30 days', '31-60 days', '61-90 days', '91-180 days', '180+ days']
    )
    
    bucket_counts = timeline_df.groupby(['Timeline_Bucket', 'Risk_Category']).size().reset_index(name='Count')
    
    fig_timeline = px.bar(
        bucket_counts, 
        x='Timeline_Bucket', 
        y='Count', 
        color='Risk_Category',
        title="Employee Departure Timeline",
        color_discrete_map={
            'High': '#d32f2f',
            'Medium': '#f57c00'
        },
        hover_data=['Count']
    )
    
    fig_timeline.update_layout(
        title={
            'text': "Employee Departure Timeline",
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
    st.markdown('<div class="main-header">🚨 HR Attrition Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Show model information if available
    if results is not None:
        with st.expander("📊 Model Performance Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Classification Model (WHO)")
                if 'best_classification_model' in results:
                    st.write(f"**Best Model:** {results['best_classification_model']}")
                if 'classification_results' in results:
                    for model, metrics in results['classification_results'].items():
                        if 'auc' in metrics:
                            st.write(f"**{model} AUC:** {metrics['auc']:.3f}")
            
            with col2:
                st.subheader("📅 Regression Model (WHEN)")
                if 'best_regression_model' in results:
                    st.write(f"**Best Model:** {results['best_regression_model']}")
                if 'regression_results' in results:
                    for model, metrics in results['regression_results'].items():
                        if 'r2' in metrics:
                            st.write(f"**{model} R²:** {metrics['r2']:.3f}")
                        if 'mae' in metrics:
                            st.write(f"**{model} MAE:** {metrics['mae']:.1f} days")
    
    # Generate predictions
    st.info("🔄 Generating predictions for active employees...")
    predictions_df = generate_predictions(employee_df, models)
    
    if predictions_df.empty:
        st.error("Unable to generate predictions. Please check your models and data.")
        return
    
    st.success(f"✅ Successfully generated predictions for {len(predictions_df)} active employees")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    departments = ['All Departments'] + sorted(predictions_df['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Filter by Department", departments)
    
    risk_levels = ['All Risk Levels', 'High', 'Medium', 'Low']
    selected_risk = st.sidebar.selectbox("Filter by Risk Level", risk_levels)
    
    time_horizons = ['All Time', 'Leaving in 30 days', 'Leaving in 60 days', 'Leaving in 90 days']
    selected_time = st.sidebar.selectbox("Filter by Timeline", time_horizons)
    
    # Apply filters
    filtered_df = predictions_df.copy()
    
    if selected_dept != 'All Departments':
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    
    if selected_risk != 'All Risk Levels':
        filtered_df = filtered_df[filtered_df['Risk_Category'] == selected_risk]
    
    if selected_time != 'All Time':
        days_filter = {'Leaving in 30 days': 30, 'Leaving in 60 days': 60, 'Leaving in 90 days': 90}
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
        (filtered_df['Predicted_Lead_Time_Days'] <= 30)
    ])
    
    with col1:
        st.metric("Total Active Employees", total_active)
    
    with col2:
        st.metric("High Risk Employees", high_risk, delta=f"{high_risk/total_active*100:.1f}%" if total_active > 0 else "0%")
    
    with col3:
        st.metric("Medium Risk Employees", medium_risk, delta=f"{medium_risk/total_active*100:.1f}%" if total_active > 0 else "0%")
    
    with col4:
        st.metric("🚨 Leaving in 30 Days", urgent_cases)
    
    # Alert for urgent cases
    if urgent_cases > 0:
        st.markdown('<div class="urgent-card">', unsafe_allow_html=True)
        st.markdown(f"### 🚨 URGENT ACTION REQUIRED")
        st.markdown(f"**{urgent_cases} high-risk employees** may leave within 30 days!")
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
                    'Attrition_Probability', 'Estimated_Departure_Date',
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
                    file_name=f'high_risk_employees_{datetime.now().strftime("%Y%m%d")}.csv',
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
                    'Attrition_Probability', 'Job_Satisfaction_Score', 'Manager_Rating'
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
            
            # Top department at risk
            if selected_dept == 'All Departments':
                dept_high_risk = filtered_df[filtered_df['Risk_Category'] == 'High']['Department'].value_counts()
                if len(dept_high_risk) > 0:
                    st.metric("Top Risk Department", dept_high_risk.index[0])
                    st.metric("High Risk Count", dept_high_risk.iloc[0])
    
    with tab2:
        st.header("📅 WHEN are they Leaving - Timeline View")
        
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
            # Timeline categories
            st.subheader("📊 Departure Timeline Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            next_30 = len(timeline_df[timeline_df['Predicted_Lead_Time_Days'] <= 30])
            next_60 = len(timeline_df[(timeline_df['Predicted_Lead_Time_Days'] > 30) & (timeline_df['Predicted_Lead_Time_Days'] <= 60)])
            next_90 = len(timeline_df[(timeline_df['Predicted_Lead_Time_Days'] > 60) & (timeline_df['Predicted_Lead_Time_Days'] <= 90)])
            beyond_90 = len(timeline_df[timeline_df['Predicted_Lead_Time_Days'] > 90])
            
            with col1:
                st.markdown('<div class="urgent-card">', unsafe_allow_html=True)
                st.metric("Next 30 Days", next_30)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.metric("31-60 Days", next_60)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.metric("61-90 Days", next_90)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="safe-card">', unsafe_allow_html=True)
                st.metric("Beyond 90 Days", beyond_90)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interactive timeline visualization
            fig_timeline = create_timeline_chart(timeline_df)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Timeline table
            st.subheader("📋 Departure Timeline")
            st.info("💡 Click on any Employee ID below to view detailed information")
            
            timeline_display_cols = [
                'Employee_ID', 'Name', 'Department', 'Risk_Category',
                'Predicted_Lead_Time_Days', 'Estimated_Departure_Date',
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
                        "Days Until Departure",
                        help="Predicted number of days until employee leaves",
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
                        st.write(f"**Days Until Departure:** {int(employee_data['Predicted_Lead_Time_Days'])}")
                    
                    with result_col3:
                        st.write(f"**Estimated Departure:** {employee_data.get('Estimated_Departure_Date', 'N/A')}")
                        
                        # Color code based on urgency
                        days_left = employee_data['Predicted_Lead_Time_Days']
                        if days_left <= 30:
                            st.error("🚨 URGENT: Leaving within 30 days!")
                        elif days_left <= 60:
                            st.warning("⚠️ WARNING: Leaving within 60 days")
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
            placeholder="e.g., EMP001, EMP123",
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
                    st.markdown("**Risk Assessment**")
                    
                    risk_category = employee_data['Risk_Category']
                    if risk_category == 'High':
                        st.markdown('<div class="urgent-card">', unsafe_allow_html=True)
                        st.write(f"**Risk Level:** 🔴 {risk_category} Risk")
                        st.write(f"**Probability:** {employee_data['Attrition_Probability']:.1%}")
                        if not pd.isna(employee_data.get('Estimated_Departure_Date')):
                            st.write(f"**Est. Departure:** {employee_data['Estimated_Departure_Date']}")
                            st.write(f"**Days Until Departure:** {int(employee_data['Predicted_Lead_Time_Days'])}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif risk_category == 'Medium':
                        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                        st.write(f"**Risk Level:** 🟡 {risk_category} Risk")
                        st.write(f"**Probability:** {employee_data['Attrition_Probability']:.1%}")
                        if not pd.isna(employee_data.get('Estimated_Departure_Date')):
                            st.write(f"**Est. Departure:** {employee_data['Estimated_Departure_Date']}")
                            st.write(f"**Days Until Departure:** {int(employee_data['Predicted_Lead_Time_Days'])}")
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
                        st.metric("Job Satisfaction", f"{employee_data['Job_Satisfaction_Score']}/10")
                    if 'Manager_Rating' in employee_data:
                        st.metric("Manager Rating", f"{employee_data['Manager_Rating']}/10")
                
                with col2:
                    if 'Market_Salary_Ratio' in employee_data:
                        st.metric("Market Salary Ratio", f"{employee_data['Market_Salary_Ratio']:.2f}")
                    if 'Monthly_Salary' in employee_data:
                        st.metric("Monthly Salary", f"${employee_data['Monthly_Salary']:,}")
                
                # Recommendations
                if risk_category in ['High', 'Medium']:
                    st.subheader("💡 Recommended Actions")
                    
                    recommendations = []
                    if employee_data.get('Job_Satisfaction_Score', 10) < 6:
                        recommendations.append("🗣️ Schedule one-on-one satisfaction discussion")
                    if employee_data.get('Market_Salary_Ratio', 1.0) < 0.9:
                        recommendations.append("💰 Review and potentially adjust compensation package")
                    if employee_data.get('Manager_Rating', 10) < 6:
                        recommendations.append("👥 Assess manager-employee relationship dynamics")
                    if not pd.isna(employee_data.get('Predicted_Lead_Time_Days')) and employee_data.get('Predicted_Lead_Time_Days', 365) <= 30:
                        recommendations.append("⚡ URGENT: Schedule immediate retention meeting")
                    
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                    else:
                        st.write("• Monitor regularly and maintain engagement")
                
            else:
                st.error(f"❌ Employee ID '{target_employee_id}' not found.")
        else:
            st.info("🔍 Please enter an Employee ID or select from the dropdown.")
    
    with tab4:
        st.header("📈 Analytics & Insights")
        
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
            # Salary ratio analysis
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
        st.subheader("🏢 Department-wise Risk Analysis")
        
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
        
        dept_analysis = filtered_df.groupby('Department').agg(analysis_cols).round(3)
        
        # Rename columns
        new_column_names = ['Total Employees', 'Avg Attrition Prob']
        if 'Job_Satisfaction_Score' in analysis_cols:
            new_column_names.append('Avg Job Satisfaction')
        if 'Manager_Rating' in analysis_cols:
            new_column_names.append('Avg Manager Rating')
        if 'Market_Salary_Ratio' in analysis_cols:
            new_column_names.append('Avg Salary Ratio')
        
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
                ) if 'Avg Salary Ratio' in new_column_names else None
            }
        )
        
        # Key insights summary
        st.subheader("🔍 Key Insights Summary")
        
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
        
        # Department with highest risk
        if len(filtered_df) > 0:
            dept_risk = filtered_df.groupby('Department')['Attrition_Probability'].mean().sort_values(ascending=False)
            if len(dept_risk) > 0:
                insights.append(f"🏢 **{dept_risk.index[0]}** department has the highest average attrition risk ({dept_risk.iloc[0]:.1%})")
        
        # Urgent cases
        urgent_count = len(filtered_df[
            (filtered_df['Risk_Category'] == 'High') & 
            (filtered_df['Predicted_Lead_Time_Days'] <= 30)
        ])
        
        if urgent_count > 0:
            insights.append(f"🚨 **{urgent_count} employees** require immediate attention (leaving within 30 days)")
        
        # Prediction coverage
        with_predictions = len(filtered_df[filtered_df['Predicted_Lead_Time_Days'].notna()])
        coverage_pct = (with_predictions / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        insights.append(f"📈 **{coverage_pct:.1f}%** of employees have departure timeline predictions")
        
        for insight in insights:
            st.markdown(insight)
        
        # Model performance summary
        if results is not None:
            st.subheader("🤖 Model Performance Summary")
            
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
                st.markdown("**Regression Model (WHEN)**")
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

if __name__ == "__main__":
    main()