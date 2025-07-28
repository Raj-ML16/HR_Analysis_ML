# setup.py - Setup script for HR Predictive Analytics Dashboard (Windows Fixed)

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "fastapi",
        "uvicorn[standard]",
        "pandas",
        "numpy", 
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "joblib",
        "openpyxl",
        "python-multipart"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            return False
    
    print("All packages installed successfully!")
    return True

def create_directory_structure():
    """Create necessary directory structure"""
    print("Creating directory structure...")
    
    directories = [
        "static",
        "templates", 
        "data",
        "exports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Directory structure created!")

def check_required_files():
    """Check if all required Python modules exist"""
    print("Checking required files...")
    
    required_files = [
        "attrition_modelling.py",
        "Hiring_calculation.py", 
        "Optimize_Simple_Calculation.py",
        "business_growth.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"Found: {file}")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        print("Please ensure all your Python modules are in the same directory as this setup script.")
        return False
    
    print("All required files found!")
    return True

def create_run_script():
    """Create a convenient run script"""
    print("Creating run script...")
    
    run_script_content = '''#!/usr/bin/env python3
"""
HR Predictive Analytics Dashboard Launcher
Run this script to start the complete application
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("Starting HR Predictive Analytics Dashboard...")
    print("=" * 60)
    
    # Check if FastAPI server file exists
    if not Path("main.py").exists():
        print("main.py (FastAPI server) not found!")
        print("Please ensure main.py is in the same directory.")
        return
    
    # Check if dashboard HTML exists
    dashboard_files = ["dashboard.html", "web_interface.html"]
    dashboard_file = None
    for file in dashboard_files:
        if Path(file).exists():
            dashboard_file = file
            break
    
    if not dashboard_file:
        print("Dashboard HTML file not found!")
        print("Please ensure dashboard.html is in the same directory.")
        return
    
    try:
        print("Starting FastAPI server...")
        print("Dashboard will be available at: http://localhost:8000")
        print("API documentation at: http://localhost:8000/docs")
        print("Opening dashboard in browser in 3 seconds...")
        
        # Start FastAPI server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
        
        # Wait a moment then open browser
        time.sleep(3)
        webbrowser.open("http://localhost:8000")
        
        print("\\nDashboard is running!")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Wait for user to stop
        process.wait()
        
    except KeyboardInterrupt:
        print("\\nShutting down dashboard...")
        process.terminate()
        process.wait()
        print("Dashboard stopped successfully!")
    except Exception as e:
        print(f"Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
'''
    
    # Use UTF-8 encoding explicitly to handle any special characters
    with open("run_dashboard.py", "w", encoding='utf-8') as f:
        f.write(run_script_content)
    
    print("Created run_dashboard.py")

def create_readme():
    """Create README with instructions"""
    print("Creating README...")
    
    readme_content = '''# HR Predictive Analytics Dashboard

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
'''
    
    # Use UTF-8 encoding explicitly
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print("Created README.md")

def main():
    """Main setup function"""
    print("HR Predictive Analytics Dashboard Setup")
    print("=" * 60)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Setup failed during package installation")
        return
    
    # Step 2: Create directory structure
    create_directory_structure()
    
    # Step 3: Check required files
    if not check_required_files():
        print("Setup incomplete - missing required files")
        return
    
    # Step 4: Create convenience scripts
    create_run_script()
    create_readme()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Replace your web_interface.html with the new dashboard.html")
    print("2. Save the FastAPI backend code as 'main.py'")
    print("3. Run: python run_dashboard.py")
    print("\nYour HR Analytics Dashboard will be ready!")
    print("Dashboard: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()