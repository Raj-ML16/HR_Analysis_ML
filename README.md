# HR Predictive Analytics Dashboard

## Overview
Complete workforce planning solution with AI-powered predictions for:
- Employee attrition forecasting (who will leave and when)
- Hiring timeline optimization 
- Resource gap elimination
- Business growth alignment

## Quick Start

### 1. Setup (First Time Only)
```bash
python setup.py
```

### 2. Start Dashboard
```bash
python run_dashboard.py
```

### 3. Access Dashboard
- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## File Structure
```
├── main.py                    # FastAPI backend server
├── dashboard.html             # Updated dashboard with API integration
├── run_dashboard.py          # Convenient launcher script
├── setup.py                  # Setup and installation script
├── attrition_modelling.py    # Module 1: ML attrition prediction
├── Hiring_calculation.py     # Module 2: Hiring timeline calculation  
├── Optimize_Simple_Calculation.py # Module 3: Timeline optimization
├── business_growth.py        # Module 4: Financial growth analysis
└── data/                     # Input data files
    ├── employee_data_processed.csv
    ├── business_data.csv
    └── financial_data.csv
```

## Manual Setup (Alternative)

### Install Dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost matplotlib seaborn joblib openpyxl python-multipart
```

### Start Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Using the Dashboard

### 1. Attrition Prediction Tab
- View high-risk employees
- Analyze risk by department
- See predicted departure timeline
- Export attrition reports

### 2. Hiring Timeline Tab  
- Review immediate hiring actions
- See replacement vs growth hires
- View hiring timeline
- Export hiring plans

### 3. Timeline Optimization Tab
- Optimize hiring start dates
- Eliminate resource gaps
- View lead times by department
- Export optimized schedules

### 4. Business Growth Tab
- Monitor financial health
- Review growth priorities
- See budget utilization
- Get strategic recommendations

## Dashboard Controls

### Main Actions
- **Run Full Analysis**: Execute all 4 modules in sequence
- **Refresh Data**: Reload latest results
- **Export All Reports**: Download all Excel reports

### Per-Tab Actions
- **Export**: Download specific report
- **Re-run**: Execute specific module
- **Filter/View**: Customize data display

## API Endpoints

### Core Endpoints
- `GET /api/attrition-predictions` - Get attrition data
- `GET /api/hiring-plan` - Get hiring timeline
- `GET /api/optimization-results` - Get optimized schedule
- `GET /api/growth-analysis` - Get growth analysis

### Action Endpoints  
- `POST /api/run-full-analysis` - Run all modules
- `POST /api/run-attrition-model` - Run Module 1
- `POST /api/run-hiring-calculation` - Run Module 2
- `POST /api/run-optimization` - Run Module 3
- `POST /api/run-growth-analysis` - Run Module 4

### Export Endpoints
- `GET /api/export/{report_type}` - Download Excel reports
- `GET /api/files` - List generated files

## Troubleshooting

### Dashboard won't load
1. Check if FastAPI server is running
2. Verify port 8000 is available
3. Check browser console for errors

### No data showing
1. Run "Run Full Analysis" first
2. Check if input CSV files exist
3. Verify Python modules are working

### Module execution fails
1. Check input data format
2. Verify all dependencies installed
3. Check console/logs for specific errors

### Export not working
1. Ensure modules have generated Excel files
2. Check file permissions
3. Try refreshing data first

## Features
- Real-time ML predictions
- Interactive visualizations  
- Excel export functionality
- Mobile-responsive design
- RESTful API integration
- Automated pipeline execution
- Professional HR interface
