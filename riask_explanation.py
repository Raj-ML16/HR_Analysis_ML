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
print("ENHANCED MACHINE LEARNING PIPELINE")
print("WITH INDIVIDUAL RISK EXPLANATIONS")
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
    
    # Verify data types
    print(f"\n📊 Data Type Verification:")
    problematic_cols = []
    for col in numeric_columns:
        if col in df_fixed.columns:
            dtype = df_fixed[col].dtype
            if dtype == 'object':
                problematic_cols.append(col)
            print(f"   {col}: {dtype}")
    
    if problematic_cols:
        print(f"❌ Still have object types: {problematic_cols}")
    else:
        print("✅ All numeric columns properly converted!")
    
    return df_fixed

# Apply data type fixes
employee_df = fix_data_types(employee_df)

# 2. INDIVIDUAL RISK EXPLANATION SYSTEM
print("\n" + "="*40)
print("STEP 2: INDIVIDUAL RISK EXPLANATION SYSTEM")
print("="*40)

class IndividualRiskExplainer:
    """Fast rule-based individual risk explanation system"""
    
    def __init__(self, global_feature_importance=None):
        self.global_importance = global_feature_importance or {}
        
    def explain_employee_risk(self, employee_data, feature_cols):
        """Generate individual risk explanations for an employee"""
        
        explanations = []
        risk_factors = []
        recommendations = []
        urgency = "Monitor"
        
        # Extract key values safely
        job_satisfaction = float(employee_data.get('Job_Satisfaction_Score', 7))
        market_salary_ratio = float(employee_data.get('Market_Salary_Ratio', 1.0))
        manager_rating = float(employee_data.get('Manager_Rating', 7))
        intent_to_stay = float(employee_data.get('Intent_To_Stay_12Months', 7))
        engagement_score = float(employee_data.get('Engagement_Survey_Score', 7))
        days_since_promotion = float(employee_data.get('Days_Since_Last_Promotion', 365))
        age = float(employee_data.get('Age', 30))
        tenure_years = float(employee_data.get('Tenure_Years', 2))
        internal_applications = float(employee_data.get('Internal_Job_Applications', 0))
        
        # Job Satisfaction Analysis
        if job_satisfaction < 4:
            explanations.append(f"😟 Very low job satisfaction ({job_satisfaction:.1f}/10) - critical risk factor")
            risk_factors.append(("Job Satisfaction", "Critical", job_satisfaction))
            recommendations.append("🆘 IMMEDIATE: Emergency retention meeting required")
            urgency = "Immediate"
        elif job_satisfaction < 6:
            explanations.append(f"😐 Below-average job satisfaction ({job_satisfaction:.1f}/10) increases risk")
            risk_factors.append(("Job Satisfaction", "High", job_satisfaction))
            recommendations.append("🗣️ Schedule satisfaction discussion within 1 week")
            if urgency == "Monitor":
                urgency = "Priority"
        elif job_satisfaction < 7.5:
            explanations.append(f"😊 Moderate job satisfaction ({job_satisfaction:.1f}/10)")
            risk_factors.append(("Job Satisfaction", "Moderate", job_satisfaction))
        
        # Salary Analysis
        if market_salary_ratio < 0.75:
            explanations.append(f"💰 Significantly underpaid (salary ratio: {market_salary_ratio:.2f}) - major risk")
            risk_factors.append(("Market Salary", "Critical", market_salary_ratio))
            recommendations.append("💰 URGENT: Immediate salary adjustment needed")
            urgency = "Immediate"
        elif market_salary_ratio < 0.85:
            explanations.append(f"💰 Below market salary (ratio: {market_salary_ratio:.2f}) drives risk")
            risk_factors.append(("Market Salary", "High", market_salary_ratio))
            recommendations.append("💰 Review and adjust compensation package")
            if urgency == "Monitor":
                urgency = "Priority"
        elif market_salary_ratio < 0.95:
            explanations.append(f"💰 Slightly below market (ratio: {market_salary_ratio:.2f})")
            risk_factors.append(("Market Salary", "Moderate", market_salary_ratio))
        
        # Manager Relationship
        if manager_rating < 4:
            explanations.append(f"👥 Poor manager relationship ({manager_rating:.1f}/10) - critical issue")
            risk_factors.append(("Manager Relationship", "Critical", manager_rating))
            recommendations.append("👥 CRITICAL: Address manager relationship immediately")
            urgency = "Immediate"
        elif manager_rating < 6:
            explanations.append(f"👥 Below-average manager rating ({manager_rating:.1f}/10)")
            risk_factors.append(("Manager Relationship", "High", manager_rating))
            recommendations.append("👥 Improve manager-employee relationship dynamics")
            if urgency == "Monitor":
                urgency = "Priority"
        
        # Retention Intent (if available)
        if intent_to_stay < 3:
            explanations.append(f"🚨 Very low retention intent ({intent_to_stay:.1f}/10) - departure likely")
            risk_factors.append(("Retention Intent", "Critical", intent_to_stay))
            recommendations.append("🚨 EMERGENCY: Employee planning to leave")
            urgency = "Immediate"
        elif intent_to_stay < 5:
            explanations.append(f"📋 Low retention intent ({intent_to_stay:.1f}/10)")
            risk_factors.append(("Retention Intent", "High", intent_to_stay))
            if urgency == "Monitor":
                urgency = "Priority"
        
        # Career Progression
        months_since_promotion = days_since_promotion / 30
        if months_since_promotion > 36:
            explanations.append(f"📈 Career stagnation ({months_since_promotion:.0f} months since promotion)")
            risk_factors.append(("Career Progress", "High", months_since_promotion))
            recommendations.append("📈 Discuss career progression and advancement opportunities")
            if urgency == "Monitor":
                urgency = "Priority"
        elif months_since_promotion > 24:
            explanations.append(f"📈 Long time since promotion ({months_since_promotion:.0f} months)")
            risk_factors.append(("Career Progress", "Moderate", months_since_promotion))
        
        # Engagement (if available)
        if engagement_score < 4:
            explanations.append(f"📊 Low engagement ({engagement_score:.1f}/10) increases risk")
            risk_factors.append(("Engagement", "High", engagement_score))
            recommendations.append("📈 Focus on engagement initiatives")
            if urgency == "Monitor":
                urgency = "Priority"
        elif engagement_score < 6:
            explanations.append(f"📊 Moderate engagement ({engagement_score:.1f}/10)")
            risk_factors.append(("Engagement", "Moderate", engagement_score))
        
        # Job Search Activity
        if internal_applications > 3:
            explanations.append(f"🔍 High internal job search activity ({int(internal_applications)} applications)")
            risk_factors.append(("Job Search Activity", "High", internal_applications))
            recommendations.append("🔍 Address career satisfaction - employee actively looking")
            if urgency == "Monitor":
                urgency = "Priority"
        elif internal_applications > 1:
            explanations.append(f"🔍 Some internal job search activity ({int(internal_applications)} applications)")
            risk_factors.append(("Job Search Activity", "Moderate", internal_applications))
        
        # Age and Tenure Patterns
        if age < 25 and tenure_years < 1:
            explanations.append(f"👤 Young employee ({int(age)}) with short tenure ({tenure_years:.1f} years) - flight risk")
            risk_factors.append(("Age/Tenure Pattern", "Moderate", age))
        elif age > 55 and tenure_years > 10:
            explanations.append(f"👤 Senior employee ({int(age)}) may consider retirement options")
            risk_factors.append(("Age/Tenure Pattern", "Moderate", age))
        
        # Default explanations if none found
        if not explanations:
            explanations.append("📊 Risk assessment based on multiple combined factors")
            risk_factors.append(("Multiple Factors", "Low", 0))
        
        if not recommendations:
            recommendations.append("📋 Continue regular monitoring and engagement initiatives")
        
        return {
            'explanations': explanations,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'urgency': urgency,
            'top_3_factors': [rf[0] for rf in sorted(risk_factors, key=lambda x: self._get_priority_score(x[1]), reverse=True)[:3]],
            'primary_risk_driver': sorted(risk_factors, key=lambda x: self._get_priority_score(x[1]), reverse=True)[0][0] if risk_factors else "Multiple Factors"
        }
    
    def _get_priority_score(self, risk_level):
        """Convert risk level to numeric score for sorting"""
        scores = {"Critical": 4, "High": 3, "Moderate": 2, "Low": 1}
        return scores.get(risk_level, 1)

# Initialize risk explainer
risk_explainer = IndividualRiskExplainer()

# 3. ORIGINAL MODEL TRAINING (with fixed data types)
print("\n" + "="*40)
print("STEP 3: MODEL TRAINING WITH FIXED DATA TYPES")
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

# WHO WILL LEAVE - Classification
print("\n📊 Training Classification Models...")
X = df_model[feature_cols]
y = df_model['Attrition_Target']

# Verify data types before training
print(f"🔍 Feature data types verification:")
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"❌ Still have object columns: {object_cols}")
    # Convert any remaining object columns
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    print("✅ Converted remaining object columns to numeric")
else:
    print("✅ All features are numeric - ready for training!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Enhanced classification results storage
classification_results = {}

def evaluate_classification_model(y_true, y_pred, y_prob, model_name):
    """Comprehensive classification evaluation"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"  {model_name} Metrics:")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    F1-Score: {f1:.3f}")
    print(f"    AUC: {auc:.3f}")
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'auc': auc
    }

# Train Random Forest (primary model)
print("🌲 Training Random Forest...")
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='roc_auc')
rf_grid.fit(X_train, y_train)

rf_pred = rf_grid.predict(X_test)
rf_prob = rf_grid.predict_proba(X_test)[:, 1]
rf_metrics = evaluate_classification_model(y_test, rf_pred, rf_prob, "Random Forest")
classification_results['Random Forest'] = {**rf_metrics, 'model': rf_grid.best_estimator_}

# Train XGBoost (backup model)
print("\n🚀 Training XGBoost...")
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

# WHEN WILL THEY LEAVE - Regression
print("\n📅 Training Regression Models...")
resigned_df = df_model[df_model['Status'] == 'Resigned'].copy()
X_reg = resigned_df[feature_cols]
y_reg = resigned_df['Lead_Time']

print(f"Regression dataset: {X_reg.shape}")
print(f"Lead time range: {y_reg.min():.0f} - {y_reg.max():.0f} days")
print(f"Lead time average: {y_reg.mean():.1f} days")

# Ensure regression features are also numeric
object_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()
if object_cols_reg:
    for col in object_cols_reg:
        X_reg[col] = pd.to_numeric(X_reg[col], errors='coerce').fillna(0)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Enhanced regression results storage
regression_results = {}

def evaluate_regression_model(y_true, y_pred, model_name):
    """Comprehensive regression evaluation"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"  {model_name} Metrics:")
    print(f"    R²: {r2:.3f}")
    print(f"    MAE: {mae:.1f} days")
    print(f"    RMSE: {rmse:.1f} days")
    print(f"    MAPE: {mape:.1f}%")
    
    return {
        'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape
    }

# Train Random Forest Regressor
print("🌲 Training Random Forest Regressor...")
rfr_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rfr_grid = GridSearchCV(RandomForestRegressor(random_state=42), rfr_params, cv=3, scoring='r2')
rfr_grid.fit(X_reg_train, y_reg_train)

rfr_pred = rfr_grid.predict(X_reg_test)
rfr_metrics = evaluate_regression_model(y_reg_test, rfr_pred, "Random Forest Regressor")
regression_results['Random Forest Regressor'] = {**rfr_metrics, 'model': rfr_grid.best_estimator_}

# Train XGBoost Regressor
print("\n🚀 Training XGBoost Regressor...")
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

# Get global feature importance for explanations
best_classifier = classification_results[best_clf_model]['model']
if hasattr(best_classifier, 'feature_importances_'):
    global_importance = dict(zip(feature_cols, best_classifier.feature_importances_))
    risk_explainer.global_importance = global_importance
    
    print(f"\n📊 Top 10 Global Feature Importance:")
    sorted_importance = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
        print(f"   {i:2d}. {feature}: {importance:.3f}")

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
print("✅ Enhanced models saved!")

# 4. ENHANCED EXCEL EXPORT WITH INDIVIDUAL RISK EXPLANATIONS
print("\n" + "="*60)
print("STEP 4: ENHANCED EXCEL EXPORT WITH RISK EXPLANATIONS")
print("="*60)

def export_ENHANCED_predictions_to_excel(models_dict, employee_df, risk_explainer):
    """Export predictions with individual risk explanations"""
    
    print("🔄 Generating enhanced predictions with individual explanations...")
    
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
    
    # Calculate total lead time
    active_employees['Predicted_Lead_Time_Days'] = [
        days_to_resign + notice_days 
        for days_to_resign, notice_days in zip(resignation_timeline_days, predicted_notice_periods)
    ]
    
    # Risk categorization
    def get_risk_category(prob):
        if prob >= 0.7: return 'High'
        elif prob >= 0.4: return 'Medium'
        else: return 'Low'
    
    active_employees['Risk_Category'] = active_employees['Attrition_Probability'].apply(get_risk_category)
    
    # ENHANCED: Generate individual risk explanations
    print("🔍 Generating individual risk explanations...")
    
    explanations_data = []
    for idx, (_, employee) in enumerate(active_employees.iterrows()):
        if idx % 50 == 0:
            print(f"   Processed {idx}/{len(active_employees)} explanations...")
        
        explanation = risk_explainer.explain_employee_risk(employee, models_dict['feature_cols'])
        
        explanations_data.append({
            'Employee_ID': employee['Employee_ID'],
            'Top_Risk_Factor_1': explanation['top_3_factors'][0] if len(explanation['top_3_factors']) > 0 else 'Multiple Factors',
            'Top_Risk_Factor_2': explanation['top_3_factors'][1] if len(explanation['top_3_factors']) > 1 else '',
            'Top_Risk_Factor_3': explanation['top_3_factors'][2] if len(explanation['top_3_factors']) > 2 else '',
            'Primary_Risk_Driver': explanation['primary_risk_driver'],
            'Risk_Explanation': ' | '.join(explanation['explanations'][:2]),  # Top 2 explanations
            'Recommended_Actions': ' | '.join(explanation['recommendations'][:2]),  # Top 2 recommendations
            'Action_Urgency': explanation['urgency'],
            'Detailed_Explanation': '\n'.join(explanation['explanations'])
        })
    
    print("✅ Individual explanations generated!")
    
    # Merge explanations with predictions
    explanations_df = pd.DataFrame(explanations_data)
    active_employees = active_employees.merge(explanations_df, on='Employee_ID', how='left')
    
    # Create Excel file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Employee_Attrition_Predictions_ENHANCED_{timestamp}.xlsx'
    
    print(f"📊 Creating ENHANCED Excel file: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: ENHANCED High Risk Employees
        high_risk = active_employees[active_employees['Risk_Category'] == 'High']
        if len(high_risk) > 0:
            high_risk_cols = [
                'Employee_ID', 'Name', 'Department', 'Designation', 
                'Attrition_Probability', 'Risk_Category', 'Action_Urgency',
                'Primary_Risk_Driver', 'Risk_Explanation', 'Recommended_Actions',
                'Predicted_Notice_Period_Days', 'Estimated_Departure_Date',
                'Job_Satisfaction_Score', 'Market_Salary_Ratio', 'Manager_Rating'
            ]
            
            # Add enhanced features if available
            if 'Intent_To_Stay_12Months' in high_risk.columns:
                high_risk_cols.extend(['Intent_To_Stay_12Months', 'Engagement_Survey_Score'])
            
            # Only include columns that exist
            available_high_risk_cols = [col for col in high_risk_cols if col in high_risk.columns]
            high_risk_data = high_risk[available_high_risk_cols].sort_values('Attrition_Probability', ascending=False)
            high_risk_data.to_excel(writer, sheet_name='High_Risk_ENHANCED', index=False)
        
        # Sheet 2: ENHANCED All Predictions
        prediction_cols = [
            'Employee_ID', 'Name', 'Department', 'Risk_Category', 'Attrition_Probability',
            'Primary_Risk_Driver', 'Action_Urgency', 'Risk_Explanation',
            'Predicted_Notice_Period_Days', 'Estimated_Departure_Date'
        ]
        available_pred_cols = [col for col in prediction_cols if col in active_employees.columns]
        all_predictions = active_employees[available_pred_cols].sort_values('Attrition_Probability', ascending=False)
        all_predictions.to_excel(writer, sheet_name='All_Predictions_ENHANCED', index=False)
        
        # Sheet 3: NEW - Individual Risk Analysis
        risk_analysis_cols = [
            'Employee_ID', 'Name', 'Department', 'Risk_Category', 'Attrition_Probability',
            'Top_Risk_Factor_1', 'Top_Risk_Factor_2', 'Top_Risk_Factor_3',
            'Detailed_Explanation', 'Recommended_Actions', 'Action_Urgency'
        ]
        available_risk_cols = [col for col in risk_analysis_cols if col in active_employees.columns]
        risk_analysis_data = active_employees[available_risk_cols].sort_values('Attrition_Probability', ascending=False)
        risk_analysis_data.to_excel(writer, sheet_name='Individual_Risk_Analysis', index=False)
        
        # Sheet 4: Action Plan for HR
        action_plan_data = []
        for urgency in ['Immediate', 'Priority', 'Monitor']:
            urgency_employees = active_employees[active_employees['Action_Urgency'] == urgency]
            for _, emp in urgency_employees.iterrows():
                action_plan_data.append({
                    'Action_Urgency': urgency,
                    'Employee_ID': emp['Employee_ID'],
                    'Name': emp.get('Name', 'N/A'),
                    'Department': emp.get('Department', 'N/A'),
                    'Risk_Probability': emp['Attrition_Probability'],
                    'Primary_Issue': emp['Primary_Risk_Driver'],
                    'Immediate_Action': emp['Recommended_Actions'].split(' | ')[0] if ' | ' in str(emp['Recommended_Actions']) else emp['Recommended_Actions'],
                    'Days_Until_Departure': emp.get('Predicted_Lead_Time_Days', 365)
                })
        
        if action_plan_data:
            action_plan_df = pd.DataFrame(action_plan_data)
            action_plan_df.to_excel(writer, sheet_name='HR_Action_Plan', index=False)
        
        # Sheet 5: Risk Factor Summary
        risk_factor_summary = []
        all_factors = []
        for _, emp in active_employees.iterrows():
            if pd.notna(emp.get('Top_Risk_Factor_1')):
                all_factors.append(emp['Top_Risk_Factor_1'])
            if pd.notna(emp.get('Top_Risk_Factor_2')):
                all_factors.append(emp['Top_Risk_Factor_2'])
            if pd.notna(emp.get('Top_Risk_Factor_3')):
                all_factors.append(emp['Top_Risk_Factor_3'])
        
        from collections import Counter
        factor_counts = Counter(all_factors)
        
        for factor, count in factor_counts.most_common():
            if factor and factor != '':
                avg_risk = active_employees[
                    (active_employees['Top_Risk_Factor_1'] == factor) |
                    (active_employees['Top_Risk_Factor_2'] == factor) |
                    (active_employees['Top_Risk_Factor_3'] == factor)
                ]['Attrition_Probability'].mean()
                
                risk_factor_summary.append({
                    'Risk_Factor': factor,
                    'Employee_Count': count,
                    'Avg_Risk_Probability': avg_risk,
                    'Priority_Level': 'High' if avg_risk > 0.6 else 'Medium' if avg_risk > 0.3 else 'Low'
                })
        
        if risk_factor_summary:
            risk_summary_df = pd.DataFrame(risk_factor_summary)
            risk_summary_df = risk_summary_df.sort_values('Avg_Risk_Probability', ascending=False)
            risk_summary_df.to_excel(writer, sheet_name='Risk_Factor_Summary', index=False)
        
        # Sheet 6: ENHANCED Summary
        summary_data = {
            'Metric': [
                'Total Active Employees',
                'High Risk Employees',
                'Medium Risk Employees', 
                'Low Risk Employees',
                'Immediate Action Required',
                'Priority Action Required',
                'Most Common Risk Factor',
                'Enhanced Features Used',
                'Best Classification Model',
                'Classification AUC',
                'Best Regression Model',
                'Regression R²',
                'Individual Explanations',
                'Approach Used'
            ],
            'Value': [
                len(active_employees),
                len(active_employees[active_employees['Risk_Category'] == 'High']),
                len(active_employees[active_employees['Risk_Category'] == 'Medium']),
                len(active_employees[active_employees['Risk_Category'] == 'Low']),
                len(active_employees[active_employees['Action_Urgency'] == 'Immediate']),
                len(active_employees[active_employees['Action_Urgency'] == 'Priority']),
                factor_counts.most_common(1)[0][0] if factor_counts else 'N/A',
                len(available_enhanced),
                best_clf_model,
                f"{classification_results[best_clf_model]['auc']:.3f}",
                best_reg_model,
                f"{regression_results[best_reg_model]['r2']:.3f}",
                'Enabled for all employees',
                'ENHANCED_WITH_INDIVIDUAL_EXPLANATIONS'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='ENHANCED_Summary', index=False)
        
        # Sheet 7: Department Risk Analysis
        dept_analysis = active_employees.groupby('Department').agg({
            'Employee_ID': 'count',
            'Attrition_Probability': 'mean',
            'Predicted_Notice_Period_Days': 'mean'
        }).round(2)
        
        dept_analysis.columns = ['Employee_Count', 'Avg_Risk_Probability', 'Avg_Notice_Period']
        dept_analysis['High_Risk_Count'] = active_employees[active_employees['Risk_Category'] == 'High'].groupby('Department').size()
        dept_analysis['High_Risk_Count'] = dept_analysis['High_Risk_Count'].fillna(0).astype(int)
        dept_analysis = dept_analysis.sort_values('Avg_Risk_Probability', ascending=False)
        dept_analysis.to_excel(writer, sheet_name='Department_Analysis')
    
    print(f"✅ ENHANCED Excel file saved: {filename}")
    print(f"📊 File includes:")
    print(f"   ✅ Individual risk explanations for all {len(active_employees)} employees")
    print(f"   ✅ Personalized recommendations")
    print(f"   ✅ Action urgency levels")
    print(f"   ✅ Risk factor analysis")
    print(f"   ✅ HR action plan")
    
    return filename

# Export enhanced predictions
print("🚀 Starting enhanced export process...")
excel_filename = export_ENHANCED_predictions_to_excel(models_to_save, df_model, risk_explainer)

# Save enhanced results summary
results_summary = {
    'classification_results': classification_results,
    'regression_results': regression_results,
    'best_classification_model': best_clf_model,
    'best_regression_model': best_reg_model,
    'enhanced_features': available_enhanced,
    'total_features': len(feature_cols),
    'approach': 'ENHANCED_WITH_INDIVIDUAL_EXPLANATIONS'
}

# Convert numpy types to native Python types for JSON serialization
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

print("✅ Enhanced results saved to 'model_results_enhanced.json'")

# 5. ENHANCED FINAL RESULTS SUMMARY
print("\n" + "="*80)
print("ENHANCED FINAL RESULTS SUMMARY WITH INDIVIDUAL EXPLANATIONS")
print("="*80)

print(f"\n🎯 CLASSIFICATION RESULTS (WHO WILL LEAVE):")
for model, results in classification_results.items():
    print(f"  {model}: AUC = {results['auc']:.3f} | F1 = {results['f1_score']:.3f} | Precision = {results['precision']:.3f} | Recall = {results['recall']:.3f}")

print(f"\n📅 REGRESSION RESULTS (WHEN - NOTICE PERIODS):")
for model, results in regression_results.items():
    print(f"  {model}: R² = {results['r2']:.3f} | RMSE = {results['rmse']:.1f} | MAE = {results['mae']:.1f} | MAPE = {results['mape']:.1f}%")

print(f"\n🏆 BEST MODELS:")
print(f"  Classification: {best_clf_model} (AUC: {classification_results[best_clf_model]['auc']:.3f})")
print(f"  Regression: {best_reg_model} (R²: {regression_results[best_reg_model]['r2']:.3f})")

print(f"\n🔍 INDIVIDUAL RISK EXPLANATIONS:")
print(f"  ✅ Rule-based explanations for all employees")
print(f"  ✅ Personalized risk factors identified")
print(f"  ✅ Specific recommendations generated")
print(f"  ✅ Action urgency levels assigned")

print(f"\n📁 ENHANCED FILES CREATED:")
print(f"  ✅ Enhanced Models: 'attrition_models_enhanced.pkl'")
print(f"  ✅ Enhanced Results: 'model_results_enhanced.json'")
print(f"  ✅ Enhanced Excel: '{excel_filename}'")

print(f"\n📊 EXCEL SHEETS INCLUDED:")
print(f"  1. High_Risk_ENHANCED - High-risk employees with explanations")
print(f"  2. All_Predictions_ENHANCED - All employees with risk drivers")
print(f"  3. Individual_Risk_Analysis - Detailed individual explanations")
print(f"  4. HR_Action_Plan - Prioritized action items for HR")
print(f"  5. Risk_Factor_Summary - Organization-wide risk patterns")
print(f"  6. ENHANCED_Summary - Model performance and approach")
print(f"  7. Department_Analysis - Department-wise risk breakdown")

print(f"\n✅ ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
print("🚀 Now includes individual risk explanations for every employee!")
print("💡 HR team can now see WHY each employee is at risk and WHAT to do about it!")
print("="*80)

# Example: Show individual explanation for a high-risk employee
active_employees = df_model[df_model['Status'] == 'Active']
if len(active_employees) > 0:
    # Get predictions for demonstration
    X_demo = active_employees[feature_cols].head(1)
    
    # Ensure numeric types
    object_cols = X_demo.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        for col in object_cols:
            X_demo[col] = pd.to_numeric(X_demo[col], errors='coerce').fillna(0)
    
    demo_prob = models_to_save['best_classifier'].predict_proba(X_demo)[0, 1]
    demo_employee = active_employees.iloc[0]
    
    print(f"\n" + "="*60)
    print("INDIVIDUAL EXPLANATION EXAMPLE")
    print("="*60)
    print(f"Employee: {demo_employee['Employee_ID']} - {demo_employee.get('Name', 'N/A')}")
    print(f"Risk Probability: {demo_prob:.1%}")
    
    demo_explanation = risk_explainer.explain_employee_risk(demo_employee, feature_cols)
    print(f"Primary Risk Driver: {demo_explanation['primary_risk_driver']}")
    print(f"Action Urgency: {demo_explanation['urgency']}")
    
    print(f"\nDetailed Explanations:")
    for i, explanation in enumerate(demo_explanation['explanations'][:3], 1):
        print(f"  {i}. {explanation}")
    
    print(f"\nRecommended Actions:")
    for i, recommendation in enumerate(demo_explanation['recommendations'][:2], 1):
        print(f"  {i}. {recommendation}")
    
    print("="*60)

print(f"\n💡 USAGE INSTRUCTIONS:")
print(f"  1. Run this enhanced script to generate models and Excel file")
print(f"  2. Share Excel file with HR team for actionable insights")  
print(f"  3. Use dashboard for interactive analysis")
print(f"  4. Models saved for dashboard integration")

print(f"\n🎯 KEY BENEFITS:")
print(f"  ✅ Fixed data type issues - no more XGBoost errors")
print(f"  ✅ Individual explanations for every employee")
print(f"  ✅ Actionable recommendations for HR team")
print(f"  ✅ Urgency-based prioritization")
print(f"  ✅ Comprehensive Excel reports")
print(f"  ✅ Dashboard-ready models")