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
import warnings
warnings.filterwarnings('ignore')

# Load processed data (UPDATED for FIXED data)
employee_df = pd.read_csv('employee_data_processed.csv')

print("="*60)
print("FIXED MACHINE LEARNING PIPELINE")
print("REALISTIC Lead Times: WHO Model → WHEN Model → Export")
print("="*60)

# 1. DATA PREPARATION
print("\n" + "="*40)
print("STEP 1: FIXED DATA PREPARATION")
print("="*40)

def prepare_enhanced_features_FIXED(df):
    """Prepare enhanced features for modeling with FIXED lead time approach"""
    
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
    
    # FIXED: Handle new Notice_Period_Type column
    if 'Notice_Period_Type' in df_model.columns:
        le_notice = LabelEncoder()
        df_model['Notice_Period_Type_Encoded'] = le_notice.fit_transform(df_model['Notice_Period_Type'].fillna('Unknown'))
        print(f"✅ Notice Period Types encoded: {len(le_notice.classes_)} types")
        # Optionally add as feature (usually not needed for prediction, but available)
        # feature_cols.append('Notice_Period_Type_Encoded')
    
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
    
    # Add notice period encoder if available
    if 'Notice_Period_Type' in df_model.columns:
        encoders['notice_type'] = le_notice
    
    return df_model, feature_cols, encoders, available_enhanced

# Prepare data with FIXED approach
df_model, feature_cols, encoders, available_enhanced = prepare_enhanced_features_FIXED(employee_df)

print(f"✅ Features prepared: {len(feature_cols)} features")
print(f"✅ Enhanced features added: {len(available_enhanced)}")
print(f"✅ Dataset shape: {df_model.shape}")
print(f"✅ Active employees: {len(df_model[df_model['Status'] == 'Active'])}")
print(f"✅ Resigned employees: {len(df_model[df_model['Status'] == 'Resigned'])}")

# FIXED: Analyze the new realistic lead times
resigned_df = df_model[df_model['Status'] == 'Resigned']
if len(resigned_df) > 0 and 'Lead_Time' in resigned_df.columns:
    print(f"\n📊 FIXED LEAD TIME ANALYSIS:")
    print(f"   Average notice period: {resigned_df['Lead_Time'].mean():.1f} days")
    print(f"   Median notice period: {resigned_df['Lead_Time'].median():.1f} days")
    print(f"   Notice range: {resigned_df['Lead_Time'].min():.0f} - {resigned_df['Lead_Time'].max():.0f} days")
    
    if 'Notice_Period_Type' in resigned_df.columns:
        notice_stats = resigned_df.groupby('Notice_Period_Type')['Lead_Time'].agg(['count', 'mean', 'std']).round(1)
        print(f"   Notice by Type:")
        for notice_type, stats in notice_stats.iterrows():
            print(f"     {notice_type}: {stats['count']} employees, avg {stats['mean']} days")

# 2. WHO WILL LEAVE - Enhanced Classification Models
print("\n" + "="*40)
print("STEP 2: WHO WILL LEAVE - ENHANCED CLASSIFICATION")
print("="*40)

X = df_model[feature_cols]
y = df_model['Attrition_Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Attrition rate in training: {y_train.mean():.2%}")

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

# 2.1 Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='roc_auc')
rf_grid.fit(X_train, y_train)

rf_pred = rf_grid.predict(X_test)
rf_prob = rf_grid.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_classification_model(y_test, rf_pred, rf_prob, "Random Forest")
classification_results['Random Forest'] = {**rf_metrics, 'model': rf_grid.best_estimator_}

# 2.2 XGBoost Classifier
print("\nTraining XGBoost Classifier...")
try:
    xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=3, scoring='roc_auc')
    xgb_grid.fit(X_train, y_train)

    xgb_pred = xgb_grid.predict(X_test)
    xgb_prob = xgb_grid.predict_proba(X_test)[:, 1]

    xgb_metrics = evaluate_classification_model(y_test, xgb_pred, xgb_prob, "XGBoost")
    classification_results['XGBoost'] = {**xgb_metrics, 'model': xgb_grid.best_estimator_}
except Exception as e:
    print(f"⚠️ XGBoost failed: {e}")

# Best classification model (using AUC as primary metric)
best_clf_model = max(classification_results.keys(), key=lambda x: classification_results[x]['auc'])
print(f"\n🏆 Best Classification Model: {best_clf_model} (AUC: {classification_results[best_clf_model]['auc']:.3f})")
print("🎯 AUC is the primary metric for imbalanced classification problems")

# Confusion Matrix for best model
plt.figure(figsize=(8, 6))
best_pred = rf_grid.predict(X_test) if best_clf_model == 'Random Forest' else xgb_grid.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_clf_model}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 3. WHEN WILL THEY LEAVE - FIXED Regression Models
print("\n" + "="*40)
print("STEP 3: WHEN WILL THEY LEAVE - FIXED REGRESSION")
print("🎯 Now predicting REALISTIC notice periods!")
print("="*40)

# Use only resigned employees for lead time prediction
resigned_df = df_model[df_model['Status'] == 'Resigned'].copy()
X_reg = resigned_df[feature_cols]
y_reg = resigned_df['Lead_Time']

print(f"Regression dataset: {X_reg.shape}")
print(f"Lead time range: {y_reg.min():.0f} - {y_reg.max():.0f} days")
print(f"Lead time average: {y_reg.mean():.1f} days")
print(f"Lead time std: {y_reg.std():.1f} days")

# FIXED: Better lead time distribution analysis
print(f"\nFIXED Lead Time Distribution:")
print(f"  Short notice (≤21 days): {len(y_reg[y_reg <= 21])} employees ({len(y_reg[y_reg <= 21])/len(y_reg)*100:.1f}%)")
print(f"  Standard notice (22-45 days): {len(y_reg[(y_reg > 21) & (y_reg <= 45)])} employees ({len(y_reg[(y_reg > 21) & (y_reg <= 45)])/len(y_reg)*100:.1f}%)")
print(f"  Long notice (>45 days): {len(y_reg[y_reg > 45])} employees ({len(y_reg[y_reg > 45])/len(y_reg)*100:.1f}%)")

# Split regression data
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

# 3.1 Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rfr_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
rfr_grid = GridSearchCV(RandomForestRegressor(random_state=42), rfr_params, cv=3, scoring='r2')
rfr_grid.fit(X_reg_train, y_reg_train)

rfr_pred = rfr_grid.predict(X_reg_test)
rfr_metrics = evaluate_regression_model(y_reg_test, rfr_pred, "Random Forest Regressor")
regression_results['Random Forest Regressor'] = {**rfr_metrics, 'model': rfr_grid.best_estimator_}

# 3.2 XGBoost Regressor
print("\nTraining XGBoost Regressor...")
try:
    xgbr_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
    xgbr_grid = GridSearchCV(XGBRegressor(random_state=42), xgbr_params, cv=3, scoring='r2')
    xgbr_grid.fit(X_reg_train, y_reg_train)

    xgbr_pred = xgbr_grid.predict(X_reg_test)
    xgbr_metrics = evaluate_regression_model(y_reg_test, xgbr_pred, "XGBoost Regressor")
    regression_results['XGBoost Regressor'] = {**xgbr_metrics, 'model': xgbr_grid.best_estimator_}
except Exception as e:
    print(f"⚠️ XGBoost Regressor failed: {e}")

# Best regression model (using R² as primary metric)
best_reg_model = max(regression_results.keys(), key=lambda x: regression_results[x]['r2'])
print(f"\n🏆 Best Regression Model: {best_reg_model} (R²: {regression_results[best_reg_model]['r2']:.3f})")
print("🎯 R² is the primary metric for notice period prediction accuracy")

# FIXED: Feature importance analysis for notice period prediction
print(f"\n📊 FIXED: Feature Importance for Notice Period Prediction:")
if hasattr(regression_results[best_reg_model]['model'], 'feature_importances_'):
    importances = regression_results[best_reg_model]['model'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Top 10 factors affecting notice period length:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']}: {row['importance']:.3f}")

# 4. SAVE FIXED MODELS
print("\n" + "="*40)
print("STEP 4: SAVING FIXED MODELS")
print("="*40)

# Package all trained components
models_to_save = {
    'best_classifier': classification_results[best_clf_model]['model'],
    'best_regressor': regression_results[best_reg_model]['model'],
    'scaler': scaler,
    'encoders': encoders,
    'feature_cols': feature_cols,
    'enhanced_features': available_enhanced
}

# Save models
joblib.dump(models_to_save, 'attrition_models_enhanced.pkl')
print("✅ FIXED models saved to 'attrition_models_enhanced.pkl'")

# Save enhanced results summary
results_summary = {
    'classification_results': classification_results,
    'regression_results': regression_results,
    'best_classification_model': best_clf_model,
    'best_regression_model': best_reg_model,
    'enhanced_features': available_enhanced,
    'total_features': len(feature_cols),
    'approach': 'FIXED_REALISTIC_LEAD_TIMES'
}

import json
with open('model_results_enhanced.json', 'w') as f:
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
    
    json.dump(results_summary_json, f, indent=2)

print("✅ FIXED results saved to 'model_results_enhanced.json'")

# 5. EXPORT FIXED PREDICTIONS TO EXCEL
print("\n" + "="*40)
print("STEP 5: EXPORTING FIXED PREDICTIONS TO EXCEL")
print("="*40)

def get_standard_notice_estimate(designation, department):
    """Get standard notice period estimate for fallback predictions"""
    if any(title in designation for title in ['Manager', 'Director', 'VP', 'CFO']):
        return 90  # Executive level
    elif any(title in designation for title in ['Tech Lead', 'Senior']):
        return 60  # Senior level  
    elif department in ['Engineering', 'Finance']:
        return 30  # Professional level
    else:
        return 14  # Standard level

def export_FIXED_predictions_to_excel(models_dict, employee_df):
    """Export FIXED WHO and WHEN predictions to Excel"""
    
    print("Making FIXED predictions for active employees...")
    
    # Get active employees
    active_employees = employee_df[employee_df['Status'] == 'Active'].copy()
    
    if len(active_employees) == 0:
        print("❌ No active employees found!")
        return None
    
    print(f"📊 Processing {len(active_employees)} active employees...")
    
    # Prepare features and make predictions
    X = active_employees[models_dict['feature_cols']]
    
    # WHO predictions
    print("🎯 Making FIXED WHO predictions...")
    attrition_prob = models_dict['best_classifier'].predict_proba(X)[:, 1]
    
    # WHEN predictions - FIXED approach
    print("📅 Making FIXED WHEN predictions (notice period length)...")
    
    # Use ML model to predict notice period for all employees
    predicted_notice_periods = models_dict['best_regressor'].predict(X)
    
    # Ensure realistic bounds
    predicted_notice_periods = np.clip(predicted_notice_periods, 7, 180)  # 1 week to 6 months
    
    # Add predictions to dataframe
    active_employees['Attrition_Probability'] = attrition_prob.round(3)
    active_employees['Predicted_Notice_Period_Days'] = predicted_notice_periods.round().astype(int)
    
    # Calculate REALISTIC departure timeline
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
    
    # Risk categorization
    def get_risk_category(prob):
        if prob >= 0.7: return 'High'
        elif prob >= 0.4: return 'Medium'
        else: return 'Low'
    
    active_employees['Risk_Category'] = active_employees['Attrition_Probability'].apply(get_risk_category)
    
    # Create Excel file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Employee_Attrition_Predictions_FIXED_{timestamp}.xlsx'
    
    print(f"📊 Creating FIXED Excel file: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: High Risk Employees
        high_risk = active_employees[active_employees['Risk_Category'] == 'High']
        if len(high_risk) > 0:
            high_risk_cols = [
                'Employee_ID', 'Name', 'Department', 'Designation', 
                'Attrition_Probability', 'Predicted_Notice_Period_Days', 
                'Estimated_Resignation_Date', 'Estimated_Departure_Date',
                'Job_Satisfaction_Score', 'Market_Salary_Ratio', 'Manager_Rating'
            ]
            # Add enhanced features if available
            if 'Intent_To_Stay_12Months' in high_risk.columns:
                high_risk_cols.extend(['Intent_To_Stay_12Months', 'Engagement_Survey_Score'])
            
            high_risk_data = high_risk[high_risk_cols].sort_values('Attrition_Probability', ascending=False)
            high_risk_data.to_excel(writer, sheet_name='High_Risk_Employees', index=False)
        
        # Sheet 2: All Predictions
        prediction_cols = [
            'Employee_ID', 'Name', 'Department', 'Risk_Category', 'Attrition_Probability',
            'Predicted_Notice_Period_Days', 'Estimated_Resignation_Date', 'Estimated_Departure_Date'
        ]
        all_predictions = active_employees[prediction_cols].sort_values('Attrition_Probability', ascending=False)
        all_predictions.to_excel(writer, sheet_name='All_Employee_Predictions', index=False)
        
        # Sheet 3: FIXED Summary
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
                'Approach Used'
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
                'FIXED_REALISTIC_LEAD_TIMES'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='FIXED_Summary', index=False)
        
        # Sheet 4: Notice Period Analysis
        notice_analysis = active_employees.groupby('Department').agg({
            'Predicted_Notice_Period_Days': ['count', 'mean', 'min', 'max'],
            'Attrition_Probability': 'mean'
        }).round(1)
        notice_analysis.columns = ['Employee_Count', 'Avg_Notice_Days', 'Min_Notice', 'Max_Notice', 'Avg_Risk_Score']
        notice_analysis.to_excel(writer, sheet_name='Notice_Period_Analysis')
    
    print(f"✅ FIXED Excel file saved: {filename}")
    return filename

# Export FIXED predictions
excel_filename = export_FIXED_predictions_to_excel(models_to_save, df_model)

# 6. FIXED FINAL RESULTS SUMMARY
print("\n" + "="*60)
print("FIXED FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n🎯 FIXED CLASSIFICATION RESULTS (WHO WILL LEAVE):")
for model, results in classification_results.items():
    print(f"  {model}:")
    print(f"    AUC: {results['auc']:.3f} | F1: {results['f1_score']:.3f} | Precision: {results['precision']:.3f} | Recall: {results['recall']:.3f}")

print(f"\n📅 FIXED REGRESSION RESULTS (REALISTIC NOTICE PERIODS):")
for model, results in regression_results.items():
    print(f"  {model}:")
    print(f"    R²: {results['r2']:.3f} | RMSE: {results['rmse']:.1f} | MAE: {results['mae']:.1f} | MAPE: {results['mape']:.1f}%")

print(f"\n📈 ENHANCED FEATURES IMPACT:")
print(f"  Enhanced features added: {len(available_enhanced)}")
print(f"  Total features: {len(feature_cols)}")

print(f"\n🏆 BEST MODELS:")
print(f"  Classification: {best_clf_model} (AUC: {classification_results[best_clf_model]['auc']:.3f}) - Primary metric")
print(f"  Regression: {best_reg_model} (R²: {regression_results[best_reg_model]['r2']:.3f}) - Primary metric")

print(f"\n📁 FILES CREATED:")
print(f"  ✅ FIXED Models: 'attrition_models_enhanced.pkl'")
print(f"  ✅ FIXED Results: 'model_results_enhanced.json'")
print(f"  ✅ FIXED Predictions: '{excel_filename}'")
print(f"\n✅ FIXED PIPELINE COMPLETED SUCCESSFULLY!")
print("🚀 Now using REALISTIC notice period predictions!")
print("="*60)