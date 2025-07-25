import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load processed data
employee_df = pd.read_csv('employee_data_processed.csv')

print("="*60)
print("FIXED MACHINE LEARNING PIPELINE")
print("Proper sequence: Data → WHO Model → WHEN Model → Export")
print("="*60)

# 1. DATA PREPARATION
print("\n" + "="*40)
print("STEP 1: DATA PREPARATION")
print("="*40)

def prepare_features(df):
    """Prepare features for modeling"""
    
    # Select features for modeling
    feature_cols = [
        'Age', 'Total_Experience', 'Performance_Rating', 'Avg_Work_Hours',
        'Monthly_Salary', 'Market_Salary_Ratio', 'Salary_Change_%', 'Bonus_Last_Year',
        'Job_Satisfaction_Score', 'Manager_Rating', 'Internal_Job_Applications',
        'Days_Since_Last_Promotion', 'Leaves_Taken_Last_Year', 'Commute_Distance_KM',
        'Team_Size', 'Tenure_Years', 'Experience_Tenure_Ratio', 'Salary_Satisfaction'
    ]
    
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
    
    # Add encoded features
    feature_cols.extend(['Department_Encoded', 'Designation_Encoded', 'Location_Encoded', 
                        'Education_Encoded', 'Project_Workload_Encoded', 'Work_Life_Balance_Encoded'])
    
    # Create target variables
    df_model['Attrition_Target'] = (df_model['Status'] == 'Resigned').astype(int)
    
    # Store encoders for later use
    encoders = {
        'department': le_dept,
        'designation': le_designation,
        'location': le_location,
        'education': le_education,
        'workload': le_workload,
        'worklife': le_worklife
    }
    
    return df_model, feature_cols, encoders

# Prepare data
df_model, feature_cols, encoders = prepare_features(employee_df)

print(f"✅ Features prepared: {len(feature_cols)} features")
print(f"✅ Dataset shape: {df_model.shape}")
print(f"✅ Active employees: {len(df_model[df_model['Status'] == 'Active'])}")
print(f"✅ Resigned employees: {len(df_model[df_model['Status'] == 'Resigned'])}")

# 2. WHO WILL LEAVE - Classification Models
print("\n" + "="*40)
print("STEP 2: WHO WILL LEAVE - CLASSIFICATION")
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

# Model results storage
classification_results = {}

# 2.1 Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='roc_auc')
rf_grid.fit(X_train, y_train)

rf_pred = rf_grid.predict(X_test)
rf_prob = rf_grid.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

classification_results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'auc': rf_auc,
    'model': rf_grid.best_estimator_
}

print(f"✅ RF - Accuracy: {rf_accuracy:.3f}, AUC: {rf_auc:.3f}")

# 2.2 XGBoost Classifier (Additional model for comparison)
print("Training XGBoost Classifier...")
try:
    xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=3, scoring='roc_auc')
    xgb_grid.fit(X_train, y_train)

    xgb_pred = xgb_grid.predict(X_test)
    xgb_prob = xgb_grid.predict_proba(X_test)[:, 1]

    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_prob)

    classification_results['XGBoost'] = {
        'accuracy': xgb_accuracy,
        'auc': xgb_auc,
        'model': xgb_grid.best_estimator_
    }
    
    print(f"✅ XGB - Accuracy: {xgb_accuracy:.3f}, AUC: {xgb_auc:.3f}")
except Exception as e:
    print(f"⚠️ XGBoost failed: {e}")

# Best classification model
best_clf_model = max(classification_results.keys(), key=lambda x: classification_results[x]['auc'])
print(f"\n🏆 Best Classification Model: {best_clf_model}")
print(f"🏆 Best AUC: {classification_results[best_clf_model]['auc']:.3f}")

# 3. WHEN WILL THEY LEAVE - Regression Models
print("\n" + "="*40)
print("STEP 3: WHEN WILL THEY LEAVE - REGRESSION")
print("="*40)

# Use only resigned employees for lead time prediction
resigned_df = df_model[df_model['Status'] == 'Resigned'].copy()
X_reg = resigned_df[feature_cols]
y_reg = resigned_df['Lead_Time']

print(f"Regression dataset: {X_reg.shape}")
print(f"Lead time range: {y_reg.min():.0f} - {y_reg.max():.0f} days")
print(f"Lead time average: {y_reg.mean():.1f} days")

# Split regression data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Regression results storage
regression_results = {}

# 3.1 Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rfr_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
rfr_grid = GridSearchCV(RandomForestRegressor(random_state=42), rfr_params, cv=3, scoring='r2')
rfr_grid.fit(X_reg_train, y_reg_train)

rfr_pred = rfr_grid.predict(X_reg_test)
rfr_r2 = r2_score(y_reg_test, rfr_pred)
rfr_mae = mean_absolute_error(y_reg_test, rfr_pred)
rfr_rmse = np.sqrt(mean_squared_error(y_reg_test, rfr_pred))

regression_results['Random Forest Regressor'] = {
    'r2': rfr_r2,
    'mae': rfr_mae,
    'rmse': rfr_rmse,
    'model': rfr_grid.best_estimator_
}

print(f"✅ RFR - R²: {rfr_r2:.3f}, MAE: {rfr_mae:.1f} days, RMSE: {rfr_rmse:.1f} days")

# 3.2 XGBoost Regressor (Additional model for comparison)
print("Training XGBoost Regressor...")
try:
    xgbr_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
    xgbr_grid = GridSearchCV(XGBRegressor(random_state=42), xgbr_params, cv=3, scoring='r2')
    xgbr_grid.fit(X_reg_train, y_reg_train)

    xgbr_pred = xgbr_grid.predict(X_reg_test)
    xgbr_r2 = r2_score(y_reg_test, xgbr_pred)
    xgbr_mae = mean_absolute_error(y_reg_test, xgbr_pred)
    xgbr_rmse = np.sqrt(mean_squared_error(y_reg_test, xgbr_pred))

    regression_results['XGBoost Regressor'] = {
        'r2': xgbr_r2,
        'mae': xgbr_mae,
        'rmse': xgbr_rmse,
        'model': xgbr_grid.best_estimator_
    }
    
    print(f"✅ XGBR - R²: {xgbr_r2:.3f}, MAE: {xgbr_mae:.1f} days, RMSE: {xgbr_rmse:.1f} days")
except Exception as e:
    print(f"⚠️ XGBoost Regressor failed: {e}")

# Best regression model
best_reg_model = max(regression_results.keys(), key=lambda x: regression_results[x]['r2'])
print(f"\n🏆 Best Regression Model: {best_reg_model}")
print(f"🏆 Best R²: {regression_results[best_reg_model]['r2']:.3f}")

# 4. SAVE TRAINED MODELS
print("\n" + "="*40)
print("STEP 4: SAVING TRAINED MODELS")
print("="*40)

# Package all trained components
models_to_save = {
    'best_classifier': classification_results[best_clf_model]['model'],
    'best_regressor': regression_results[best_reg_model]['model'],
    'scaler': scaler,
    'encoders': encoders,
    'feature_cols': feature_cols
}

# Save models
joblib.dump(models_to_save, 'attrition_models_fixed.pkl')
print("✅ Models saved to 'attrition_models_fixed.pkl'")

# Save results summary
results_summary = {
    'classification_results': classification_results,
    'regression_results': regression_results,
    'best_classification_model': best_clf_model,
    'best_regression_model': best_reg_model
}

import json
with open('model_results_fixed.json', 'w') as f:
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

print("✅ Results saved to 'model_results_fixed.json'")

# 5. EXPORT PREDICTIONS TO EXCEL (After models are trained!)
print("\n" + "="*40)
print("STEP 5: EXPORTING PREDICTIONS TO EXCEL")
print("Using trained models to make predictions")
print("="*40)

def estimate_departure_timeline(attrition_prob, employee_id):
    """Estimate departure timeline based on risk probability with variation"""
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

def export_predictions_to_excel(models_dict, employee_df):
    """Export WHO and WHEN predictions to Excel for HR team - UNIFIED THRESHOLDS"""
    
    print("Making predictions for active employees...")
    
    # Get active employees
    active_employees = employee_df[employee_df['Status'] == 'Active'].copy()
    
    if len(active_employees) == 0:
        print("❌ No active employees found!")
        return None
    
    print(f"📊 Processing {len(active_employees)} active employees...")
    
    # Prepare features and make predictions
    X = active_employees[models_dict['feature_cols']]
    
    # WHO predictions using trained classifier
    print("🎯 Making WHO predictions (attrition probability)...")
    attrition_prob = models_dict['best_classifier'].predict_proba(X)[:, 1]
    
    # WHEN predictions using trained regressor
    print("📅 Making WHEN predictions (departure timeline)...")
    lead_times = np.full(len(active_employees), np.nan)
    
    # ===================================================
    # UNIFIED THRESHOLDS - Same for printing AND categorization
    # ===================================================
    HIGH_THRESHOLD = 0.7    # 70%
    MEDIUM_THRESHOLD = 0.4  # 40%
    LOW_THRESHOLD = 0.2     # 20%
    
    # High Risk (>= 70%): Use ML model
    high_risk_mask = attrition_prob >= HIGH_THRESHOLD
    high_risk_count = np.sum(high_risk_mask)
    
    if high_risk_count > 0:
        high_risk_indices = np.where(high_risk_mask)[0]
        ml_predictions = models_dict['best_regressor'].predict(X.iloc[high_risk_indices])
        lead_times[high_risk_indices] = np.round(ml_predictions).astype(int)
        print(f"  🔴 {high_risk_count} high-risk employees (>= 70%): ML predictions")
    
    # Medium Risk (40-70%): Rule-based estimation
    medium_risk_mask = (attrition_prob >= MEDIUM_THRESHOLD) & (attrition_prob < HIGH_THRESHOLD)
    medium_risk_count = np.sum(medium_risk_mask)
    
    if medium_risk_count > 0:
        medium_risk_indices = np.where(medium_risk_mask)[0]
        for idx in medium_risk_indices:
            employee_id = active_employees.iloc[idx]['Employee_ID']
            lead_times[idx] = estimate_departure_timeline(attrition_prob[idx], employee_id)
        print(f"  🟡 {medium_risk_count} medium-risk employees (40-70%): Risk-based estimates")
    
    # Low Risk (20-40%): Conservative estimation
    low_risk_mask = (attrition_prob >= LOW_THRESHOLD) & (attrition_prob < MEDIUM_THRESHOLD)
    low_risk_count = np.sum(low_risk_mask)
    
    if low_risk_count > 0:
        low_risk_indices = np.where(low_risk_mask)[0]
        for idx in low_risk_indices:
            employee_id = active_employees.iloc[idx]['Employee_ID']
            lead_times[idx] = estimate_departure_timeline(attrition_prob[idx], employee_id)
        print(f"  🟢 {low_risk_count} low-risk employees (20-40%): Conservative estimates")
    
    # Add predictions to dataframe
    active_employees['Attrition_Probability'] = attrition_prob.round(3)
    active_employees['Predicted_Lead_Time_Days'] = lead_times
    
    # Calculate departure date
    today = datetime.now()
    active_employees['Estimated_Departure_Date'] = active_employees.apply(
        lambda row: (today + timedelta(days=int(row['Predicted_Lead_Time_Days']))).strftime('%Y-%m-%d') 
        if not pd.isna(row['Predicted_Lead_Time_Days']) else None, axis=1
    )
    
    # Risk categorization - SAME THRESHOLDS
    def get_risk_category(prob):
        if prob >= HIGH_THRESHOLD: return 'High'      # 70%
        elif prob >= MEDIUM_THRESHOLD: return 'Medium' # 40%
        else: return 'Low'
    
    active_employees['Risk_Category'] = active_employees['Attrition_Probability'].apply(get_risk_category)
    
    # Rest of the function stays the same...
    # (Excel creation, display results, etc.)
    
    # Create Excel file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Employee_Attrition_Predictions_FINAL_{timestamp}.xlsx'
    
    print(f"📊 Creating Excel file: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: High Risk Employees (Most Important)
        high_risk = active_employees[active_employees['Risk_Category'] == 'High']
        if len(high_risk) > 0:
            high_risk_cols = [
                'Employee_ID', 'Name', 'Department', 'Designation', 
                'Attrition_Probability', 'Predicted_Lead_Time_Days', 'Estimated_Departure_Date',
                'Job_Satisfaction_Score', 'Market_Salary_Ratio', 'Manager_Rating', 'Monthly_Salary'
            ]
            high_risk_data = high_risk[high_risk_cols].sort_values('Attrition_Probability', ascending=False)
            high_risk_data.to_excel(writer, sheet_name='High_Risk_Employees', index=False)
            print(f"  📋 High Risk sheet: {len(high_risk)} employees")
        
        # Sheet 2: All Predictions
        prediction_cols = [
            'Employee_ID', 'Name', 'Department', 'Risk_Category', 'Attrition_Probability',
            'Predicted_Lead_Time_Days', 'Estimated_Departure_Date', 
            'Job_Satisfaction_Score', 'Manager_Rating', 'Market_Salary_Ratio'
        ]
        all_predictions = active_employees[prediction_cols].sort_values('Attrition_Probability', ascending=False)
        all_predictions.to_excel(writer, sheet_name='All_Employee_Predictions', index=False)
        print(f"  📋 All Predictions sheet: {len(all_predictions)} employees")
        
        # Sheet 3: Summary
        summary_data = {
            'Metric': [
                'Total Active Employees',
                'High Risk Employees',
                'Medium Risk Employees',
                'Low Risk Employees',
                'Employees with Departure Dates',
                'Best Classification Model',
                'Classification AUC',
                'Best Regression Model',
                'Regression R²'
            ],
            'Value': [
                len(active_employees),
                len(active_employees[active_employees['Risk_Category'] == 'High']),
                len(active_employees[active_employees['Risk_Category'] == 'Medium']),
                len(active_employees[active_employees['Risk_Category'] == 'Low']),
                len(active_employees[active_employees['Estimated_Departure_Date'].notna()]),
                best_clf_model,
                f"{classification_results[best_clf_model]['auc']:.3f}",
                best_reg_model,
                f"{regression_results[best_reg_model]['r2']:.3f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"  📋 Summary sheet created")
    
    # Display results
    print(f"\n📈 RISK DISTRIBUTION:")
    risk_counts = active_employees['Risk_Category'].value_counts()
    for risk, count in risk_counts.items():
        percentage = (count / len(active_employees)) * 100
        print(f"  {risk} Risk: {count} employees ({percentage:.1f}%)")
    
    # Show departure date coverage
    with_dates = len(active_employees[active_employees['Estimated_Departure_Date'].notna()])
    without_dates = len(active_employees[active_employees['Estimated_Departure_Date'].isna()])
    print(f"\n📅 DEPARTURE DATE COVERAGE:")
    print(f"  With departure dates: {with_dates} employees ({with_dates/len(active_employees)*100:.1f}%)")
    print(f"  Without departure dates: {without_dates} employees ({without_dates/len(active_employees)*100:.1f}%)")
    
    print(f"\n✅ Excel file saved: {filename}")
    return filename

# NOW call the export function with trained models
excel_filename = export_predictions_to_excel(models_to_save, df_model)

# 6. FINAL RESULTS SUMMARY
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print(f"\n🎯 CLASSIFICATION RESULTS (WHO WILL LEAVE):")
for model, results in classification_results.items():
    print(f"  {model}: Accuracy = {results['accuracy']:.3f}, AUC = {results['auc']:.3f}")

print(f"\n📅 REGRESSION RESULTS (WHEN WILL THEY LEAVE):")
for model, results in regression_results.items():
    print(f"  {model}: R² = {results['r2']:.3f}, MAE = {results['mae']:.1f} days")

print(f"\n📁 FILES CREATED:")
print(f"  ✅ Models: 'attrition_models_fixed.pkl'")
print(f"  ✅ Results: 'model_results_fixed.json'")
print(f"  ✅ Predictions: '{excel_filename}'")

print(f"\n🏆 BEST MODELS:")
print(f"  Classification: {best_clf_model} (AUC: {classification_results[best_clf_model]['auc']:.3f})")
print(f"  Regression: {best_reg_model} (R²: {regression_results[best_reg_model]['r2']:.3f})")

print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)