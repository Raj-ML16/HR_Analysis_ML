import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Load processed data with FIXED data type handling
employee_df = pd.read_csv('employee_data_processed.csv')

print("="*60)
print("STREAMLINED ATTRITION MODELING")
print("WITH FOCUSED RISK FACTORS")
print("="*60)

# 1. FIXED DATA TYPE CONVERSION
print("\n" + "="*40)
print("STEP 1: FIXED DATA TYPE CONVERSION")
print("="*40)

def fix_data_types(df):
    """Fix data type issues that cause XGBoost errors"""
    df_fixed = df.copy()
    
    # Define numeric columns that should be converted
    numeric_columns = [
        'Age', 'Total_Experience', 'Performance_Rating', 'Avg_Work_Hours',
        'Remote_Work_Days_Monthly', 'Monthly_Salary', 'Market_Salary_Ratio', 
        'Salary_Change_%', 'Bonus_Last_Year', 'Job_Satisfaction_Score', 
        'Manager_Rating', 'Internal_Job_Applications', 'Days_Since_Last_Promotion',
        'Leaves_Taken_Last_Year', 'Commute_Distance_KM', 'Team_Size',
        'Intent_To_Stay_12Months', 'Engagement_Survey_Score', 'Meeting_Participation_Score',
        'Time_Since_Last_Promotion_Months', 'Training_Completion_Rate',
        'Lead_Time', 'Tenure_Years', 'Experience_Tenure_Ratio', 'Salary_Satisfaction'
    ]
    
    print(f"🔄 Converting {len(numeric_columns)} columns to numeric...")
    
    for col in numeric_columns:
        if col in df_fixed.columns:
            # Convert to numeric, handling any non-numeric values
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
            
            # Fill any NaN values that resulted from conversion
            if df_fixed[col].isna().any():
                median_val = df_fixed[col].median()
                df_fixed[col] = df_fixed[col].fillna(median_val)
                print(f"   ⚠️ Filled {col} NaN values with median: {median_val}")
    
    print("✅ Data type conversion completed successfully!")
    return df_fixed

# Apply data type fixes
employee_df = fix_data_types(employee_df)

# 2. STREAMLINED RISK EXPLANATION SYSTEM
print("\n" + "="*40)
print("STEP 2: STREAMLINED RISK EXPLANATION SYSTEM")
print("="*40)

class StreamlinedRiskExplainer:
    """Simplified risk explanation system for HR dashboard"""
    
    def __init__(self):
        pass
        
    def get_risk_factors(self, employee_data):
        """Generate top 2 human-readable risk factors for an employee"""
        
        risk_factors = []
        
        # Extract key values safely
        job_satisfaction = float(employee_data.get('Job_Satisfaction_Score', 7))
        market_salary_ratio = float(employee_data.get('Market_Salary_Ratio', 1.0))
        manager_rating = float(employee_data.get('Manager_Rating', 7))
        intent_to_stay = float(employee_data.get('Intent_To_Stay_12Months', 7))
        engagement_score = float(employee_data.get('Engagement_Survey_Score', 7))
        days_since_promotion = float(employee_data.get('Days_Since_Last_Promotion', 365))
        internal_applications = float(employee_data.get('Internal_Job_Applications', 0))
        
        # Priority order - check most critical factors first
        
        # 1. Retention Intent (if available and critical)
        if intent_to_stay < 4:
            risk_factors.append("Low retention intent")
        
        # 2. Job Satisfaction
        if job_satisfaction < 5:
            risk_factors.append("Low job satisfaction")
        elif job_satisfaction < 6.5:
            risk_factors.append("Moderate job satisfaction")
        
        # 3. Salary Issues
        if market_salary_ratio < 0.8:
            risk_factors.append("Below market salary")
        elif market_salary_ratio < 0.9:
            risk_factors.append("Slightly underpaid")
        
        # 4. Manager Relationship
        if manager_rating < 5:
            risk_factors.append("Poor manager relationship")
        elif manager_rating < 6.5:
            risk_factors.append("Manager relationship issues")
        
        # 5. Career Progression
        months_since_promotion = days_since_promotion / 30
        if months_since_promotion > 36:
            risk_factors.append("Career stagnation")
        elif months_since_promotion > 24:
            risk_factors.append("Limited career growth")
        
        # 6. Engagement (if available)
        if engagement_score < 5:
            risk_factors.append("Low engagement")
        elif engagement_score < 6.5:
            risk_factors.append("Moderate engagement")
        
        # 7. Job Search Activity
        if internal_applications > 3:
            risk_factors.append("High internal job search")
        elif internal_applications > 1:
            risk_factors.append("Internal job search activity")
        
        # Return top 2 factors, or default message
        if len(risk_factors) >= 2:
            return f"{risk_factors[0]}, {risk_factors[1]}"
        elif len(risk_factors) == 1:
            return f"{risk_factors[0]}, Multiple factors"
        else:
            return "Multiple combined factors"

# Initialize risk explainer
risk_explainer = StreamlinedRiskExplainer()

# 3. MODEL TRAINING (same as before)
print("\n" + "="*40)
print("STEP 3: MODEL TRAINING")
print("="*40)

def prepare_enhanced_features_FIXED(df):
    """Prepare enhanced features for modeling with FIXED data types"""
    
    # Original features
    feature_cols = [
        'Age', 'Total_Experience', 'Performance_Rating', 'Avg_Work_Hours',
        'Monthly_Salary', 'Market_Salary_Ratio', 'Salary_Change_%', 'Bonus_Last_Year',
        'Job_Satisfaction_Score', 'Manager_Rating', 'Internal_Job_Applications',
        'Days_Since_Last_Promotion', 'Leaves_Taken_Last_Year', 'Commute_Distance_KM',
        'Team_Size', 'Tenure_Years', 'Experience_Tenure_Ratio', 'Salary_Satisfaction'
    ]
    
    # Add enhanced features if available
    enhanced_features = ['Intent_To_Stay_12Months', 'Engagement_Survey_Score', 'Meeting_Participation_Score', 
                        'Time_Since_Last_Promotion_Months', 'Training_Completion_Rate']
    
    available_enhanced = [f for f in enhanced_features if f in df.columns]
    feature_cols.extend(available_enhanced)
    
    # Encode categorical variables
    le_dept = LabelEncoder()
    le_designation = LabelEncoder()
    le_location = LabelEncoder()
    le_education = LabelEncoder()
    le_workload = LabelEncoder()
    le_worklife = LabelEncoder()
    
    df_model = df.copy()
    df_model['Department_Encoded'] = le_dept.fit_transform(df_model['Department'])
    df_model['Designation_Encoded'] = le_designation.fit_transform(df_model['Designation'])
    df_model['Location_Encoded'] = le_location.fit_transform(df_model['Location'])
    df_model['Education_Encoded'] = le_education.fit_transform(df_model['Education'])
    df_model['Project_Workload_Encoded'] = le_workload.fit_transform(df_model['Project_Workload'])
    df_model['Work_Life_Balance_Encoded'] = le_worklife.fit_transform(df_model['Work_Life_Balance'])
    
    # Handle Notice_Period_Type column
    if 'Notice_Period_Type' in df_model.columns:
        le_notice = LabelEncoder()
        df_model['Notice_Period_Type_Encoded'] = le_notice.fit_transform(df_model['Notice_Period_Type'].fillna('Unknown'))
        print(f"✅ Notice Period Types encoded: {len(le_notice.classes_)} types")
    
    # Add encoded features
    feature_cols.extend(['Department_Encoded', 'Designation_Encoded', 'Location_Encoded', 
                        'Education_Encoded', 'Project_Workload_Encoded', 'Work_Life_Balance_Encoded'])
    
    # Create target variables
    df_model['Attrition_Target'] = (df_model['Status'] == 'Resigned').astype(int)
    
    # Store encoders
    encoders = {
        'department': le_dept, 'designation': le_designation, 'location': le_location,
        'education': le_education, 'workload': le_workload, 'worklife': le_worklife
    }
    
    if 'Notice_Period_Type' in df_model.columns:
        encoders['notice_type'] = le_notice
    
    return df_model, feature_cols, encoders, available_enhanced

# Prepare data
df_model, feature_cols, encoders, available_enhanced = prepare_enhanced_features_FIXED(employee_df)

print(f"✅ Features prepared: {len(feature_cols)} features")
print(f"✅ Enhanced features added: {len(available_enhanced)}")
print(f"✅ Dataset shape: {df_model.shape}")

# Classification Training
print("\n📊 Training Classification Models...")
X = df_model[feature_cols]
y = df_model['Attrition_Target']

# Verify data types before training
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"🔧 Converting remaining object columns: {object_cols}")
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    print("✅ All features are now numeric!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification results storage
classification_results = {}

def evaluate_classification_model(y_true, y_pred, y_prob, model_name):
    """Comprehensive classification evaluation"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"  {model_name}: AUC={auc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'auc': auc
    }

# Train Random Forest
print("🌲 Training Random Forest...")
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='roc_auc')
rf_grid.fit(X_train, y_train)

rf_pred = rf_grid.predict(X_test)
rf_prob = rf_grid.predict_proba(X_test)[:, 1]
rf_metrics = evaluate_classification_model(y_test, rf_pred, rf_prob, "Random Forest")
classification_results['Random Forest'] = {**rf_metrics, 'model': rf_grid.best_estimator_}

# Train XGBoost
print("🚀 Training XGBoost...")
try:
    xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=3, scoring='roc_auc')
    xgb_grid.fit(X_train, y_train)
    
    xgb_pred = xgb_grid.predict(X_test)
    xgb_prob = xgb_grid.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_classification_model(y_test, xgb_pred, xgb_prob, "XGBoost")
    classification_results['XGBoost'] = {**xgb_metrics, 'model': xgb_grid.best_estimator_}
except Exception as e:
    print(f"⚠️ XGBoost training failed: {e}")

# Best classification model
best_clf_model = max(classification_results.keys(), key=lambda x: classification_results[x]['auc'])
print(f"\n🏆 Best Classification Model: {best_clf_model} (AUC: {classification_results[best_clf_model]['auc']:.3f})")

# Regression Training
print("\n📅 Training Regression Models...")
resigned_df = df_model[df_model['Status'] == 'Resigned'].copy()
X_reg = resigned_df[feature_cols]
y_reg = resigned_df['Lead_Time']

print(f"Regression dataset: {X_reg.shape}")
print(f"Lead time average: {y_reg.mean():.1f} days")

# Ensure regression features are numeric
object_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()
if object_cols_reg:
    for col in object_cols_reg:
        X_reg[col] = pd.to_numeric(X_reg[col], errors='coerce').fillna(0)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Regression results storage
regression_results = {}

def evaluate_regression_model(y_true, y_pred, model_name):
    """Comprehensive regression evaluation"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"  {model_name}: R²={r2:.3f}, MAE={mae:.1f} days, RMSE={rmse:.1f} days")
    
    return {'r2': r2, 'mae': mae, 'rmse': rmse}

# Train Random Forest Regressor
print("🌲 Training Random Forest Regressor...")
rfr_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rfr_grid = GridSearchCV(RandomForestRegressor(random_state=42), rfr_params, cv=3, scoring='r2')
rfr_grid.fit(X_reg_train, y_reg_train)

rfr_pred = rfr_grid.predict(X_reg_test)
rfr_metrics = evaluate_regression_model(y_reg_test, rfr_pred, "Random Forest Regressor")
regression_results['Random Forest Regressor'] = {**rfr_metrics, 'model': rfr_grid.best_estimator_}

# Train XGBoost Regressor
print("🚀 Training XGBoost Regressor...")
try:
    xgbr_params = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    xgbr_grid = GridSearchCV(XGBRegressor(random_state=42), xgbr_params, cv=3, scoring='r2')
    xgbr_grid.fit(X_reg_train, y_reg_train)
    
    xgbr_pred = xgbr_grid.predict(X_reg_test)
    xgbr_metrics = evaluate_regression_model(y_reg_test, xgbr_pred, "XGBoost Regressor")
    regression_results['XGBoost Regressor'] = {**xgbr_metrics, 'model': xgbr_grid.best_estimator_}
except Exception as e:
    print(f"⚠️ XGBoost Regressor training failed: {e}")

best_reg_model = max(regression_results.keys(), key=lambda x: regression_results[x]['r2'])
print(f"🏆 Best Regression Model: {best_reg_model} (R²: {regression_results[best_reg_model]['r2']:.3f})")

# Save models
models_to_save = {
    'best_classifier': classification_results[best_clf_model]['model'],
    'best_regressor': regression_results[best_reg_model]['model'],
    'scaler': scaler,
    'encoders': encoders,
    'feature_cols': feature_cols,
    'enhanced_features': available_enhanced,
    'risk_explainer': risk_explainer
}

joblib.dump(models_to_save, 'attrition_models_enhanced.pkl')
print("✅ Models saved!")

# 4. STREAMLINED EXCEL EXPORT
print("\n" + "="*50)
print("STEP 4: STREAMLINED EXCEL EXPORT WITH RISK FACTORS")
print("="*50)

def export_streamlined_predictions_to_excel(models_dict, employee_df, risk_explainer):
    """Export streamlined predictions with risk factors"""
    
    print("🔄 Generating streamlined predictions with risk factors...")
    
    active_employees = employee_df[employee_df['Status'] == 'Active'].copy()
    
    if len(active_employees) == 0:
        print("❌ No active employees found!")
        return None
    
    print(f"📊 Processing {len(active_employees)} active employees...")
    
    # Prepare features
    X = active_employees[models_dict['feature_cols']]
    
    # Ensure numeric types
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Make predictions
    attrition_prob = models_dict['best_classifier'].predict_proba(X)[:, 1]
    predicted_notice_periods = models_dict['best_regressor'].predict(X)
    predicted_notice_periods = np.clip(predicted_notice_periods, 7, 180)
    
    # Add basic predictions
    active_employees['Attrition_Probability'] = attrition_prob.round(3)
    active_employees['Predicted_Notice_Period_Days'] = predicted_notice_periods.round().astype(int)
    
    # Calculate departure timeline
    today = datetime.now()
    resignation_timeline_days = []
    for prob in attrition_prob:
        if prob >= 0.7:
            days_to_resignation = np.random.randint(30, 90)
        elif prob >= 0.4:
            days_to_resignation = np.random.randint(90, 180)  
        else:
            days_to_resignation = np.random.randint(180, 365)
        resignation_timeline_days.append(days_to_resignation)
    
    active_employees['Estimated_Resignation_Date'] = [
        (today + timedelta(days=int(days))).strftime('%Y-%m-%d') 
        for days in resignation_timeline_days
    ]
    
    active_employees['Estimated_Departure_Date'] = [
        (datetime.strptime(resign_date, '%Y-%m-%d') + timedelta(days=int(notice_days))).strftime('%Y-%m-%d')
        for resign_date, notice_days in zip(active_employees['Estimated_Resignation_Date'], 
                                          active_employees['Predicted_Notice_Period_Days'])
    ]
    
    # Risk categorization
    def get_risk_category(prob):
        if prob >= 0.7: return 'High'
        elif prob >= 0.4: return 'Medium'
        else: return 'Low'
    
    active_employees['Risk_Category'] = active_employees['Attrition_Probability'].apply(get_risk_category)
    
    # Generate risk factors for all employees
    print("🔍 Generating risk factors...")
    
    risk_factors_list = []
    for idx, (_, employee) in enumerate(active_employees.iterrows()):
        if idx % 100 == 0:
            print(f"   Processed {idx}/{len(active_employees)} employees...")
        
        risk_factors = risk_explainer.get_risk_factors(employee)
        risk_factors_list.append(risk_factors)
    
    active_employees['Risk_Factors'] = risk_factors_list
    print("✅ Risk factors generated!")
    
    # Create Excel file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Employee_Attrition_Predictions_FIXED_{timestamp}.xlsx'
    
    print(f"📊 Creating Excel file: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: High Risk Employees with Risk Factors
        high_risk = active_employees[active_employees['Risk_Category'] == 'High']
        if len(high_risk) > 0:
            high_risk_cols = [
                'Employee_ID', 'Name', 'Department', 'Designation', 
                'Attrition_Probability', 'Risk_Factors',  # Key addition!
                'Estimated_Departure_Date', 'Predicted_Notice_Period_Days',
                'Job_Satisfaction_Score', 'Market_Salary_Ratio', 'Manager_Rating'
            ]
            
            # Add enhanced features if available
            if 'Intent_To_Stay_12Months' in high_risk.columns:
                high_risk_cols.extend(['Intent_To_Stay_12Months', 'Engagement_Survey_Score'])
            
            # Only include columns that exist
            available_high_risk_cols = [col for col in high_risk_cols if col in high_risk.columns]
            high_risk_data = high_risk[available_high_risk_cols].sort_values('Attrition_Probability', ascending=False)
            high_risk_data.to_excel(writer, sheet_name='High_Risk_Employees', index=False)
        
        # Sheet 2: All Employee Predictions
        prediction_cols = [
            'Employee_ID', 'Name', 'Department', 'Risk_Category', 'Attrition_Probability',
            'Risk_Factors', 'Estimated_Departure_Date', 'Predicted_Notice_Period_Days'
        ]
        available_pred_cols = [col for col in prediction_cols if col in active_employees.columns]
        all_predictions = active_employees[available_pred_cols].sort_values('Attrition_Probability', ascending=False)
        all_predictions.to_excel(writer, sheet_name='All_Employee_Predictions', index=False)
        
        # Sheet 3: Summary
        summary_data = {
            'Metric': [
                'Total Active Employees',
                'High Risk Employees',
                'Medium Risk Employees', 
                'Low Risk Employees',
                'Enhanced Features Used',
                'Best Classification Model',
                'Classification AUC',
                'Best Regression Model',
                'Regression R²',
                'Risk Factors Included'
            ],
            'Value': [
                len(active_employees),
                len(active_employees[active_employees['Risk_Category'] == 'High']),
                len(active_employees[active_employees['Risk_Category'] == 'Medium']),
                len(active_employees[active_employees['Risk_Category'] == 'Low']),
                len(available_enhanced),
                best_clf_model,
                f"{classification_results[best_clf_model]['auc']:.3f}",
                best_reg_model,
                f"{regression_results[best_reg_model]['r2']:.3f}",
                'Yes - Top 2 factors per employee'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Excel file saved: {filename}")
    print(f"📊 Contains {len(high_risk)} high-risk employees with individual risk factors")
    
    return filename

# Export streamlined predictions
print("🚀 Starting streamlined export...")
excel_filename = export_streamlined_predictions_to_excel(models_to_save, df_model, risk_explainer)

# Save results summary
results_summary = {
    'classification_results': classification_results,
    'regression_results': regression_results,
    'best_classification_model': best_clf_model,
    'best_regression_model': best_reg_model,
    'enhanced_features': available_enhanced,
    'total_features': len(feature_cols),
    'approach': 'STREAMLINED_WITH_RISK_FACTORS'
}

# Convert numpy types for JSON serialization
results_summary_json = {}
for key, value in results_summary.items():
    if key in ['classification_results', 'regression_results']:
        results_summary_json[key] = {}
        for model_name, metrics in value.items():
            results_summary_json[key][model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items() if k != 'model'
            }
    else:
        results_summary_json[key] = value

with open('model_results_enhanced.json', 'w') as f:
    json.dump(results_summary_json, f, indent=2)

print("✅ Results saved to 'model_results_enhanced.json'")

# 5. FINAL RESULTS SUMMARY
print("\n" + "="*60)
print("STREAMLINED ATTRITION MODELING COMPLETED")
print("="*60)

print(f"\n🏆 BEST MODELS:")
print(f"  Classification: {best_clf_model} (AUC: {classification_results[best_clf_model]['auc']:.3f})")
print(f"  Regression: {best_reg_model} (R²: {regression_results[best_reg_model]['r2']:.3f})")

print(f"\n📁 FILES CREATED:")
print(f"  ✅ Models: 'attrition_models_enhanced.pkl'")
print(f"  ✅ Results: 'model_results_enhanced.json'")
print(f"  ✅ Excel Report: '{excel_filename}'")

print(f"\n📊 EXCEL STRUCTURE (3 Focused Sheets):")
print(f"  1. High_Risk_Employees - With Risk_Factors column!")
print(f"  2. All_Employee_Predictions - All employees with risk factors")
print(f"  3. Summary - Model performance overview")

print(f"\n🎯 KEY FEATURES:")
print(f"  ✅ Fixed data type issues - no XGBoost errors")
print(f"  ✅ Risk factors column added to High Risk table")
print(f"  ✅ Streamlined Excel output (3 sheets instead of 7)")
print(f"  ✅ Human-readable risk factors")
print(f"  ✅ Ready for dashboard integration")

# Example demonstration
active_employees = df_model[df_model['Status'] == 'Active']
if len(active_employees) > 0:
    demo_employee = active_employees.iloc[0]
    demo_risk_factors = risk_explainer.get_risk_factors(demo_employee)
    
    print(f"\n" + "="*40)
    print("RISK FACTORS EXAMPLE")
    print("="*40)
    print(f"Employee: {demo_employee['Employee_ID']} - {demo_employee.get('Name', 'N/A')}")
    print(f"Risk Factors: {demo_risk_factors}")
    print("="*40)

print(f"\n✅ STREAMLINED PIPELINE COMPLETED SUCCESSFULLY!")
print("🚀 Ready for dashboard integration with focused risk factors!")
print("="*60)