#!/usr/bin/env python3
"""
HR Predictive Analytics Dashboard Launcher
Run this script to start the complete application
"""

import subprocess
import sys
import time
import webbrowser
import glob
from pathlib import Path

def check_data_files():
    """Check if analysis data files exist"""
    data_files = {
        "Attrition Model": "Employee_Attrition_Predictions_FIXED_*.xlsx",
        "Hiring Calculator": "FIXED_integrated_hiring_plan_*.xlsx", 
        "Timeline Optimizer": "SIMPLIFIED_hiring_timeline_*.xlsx",
        "Business Growth": "Enhanced_Financial_Hiring_Plan_FIXED_*.xlsx"
    }
    
    missing_files = []
    available_files = []
    
    for module_name, pattern in data_files.items():
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
            available_files.append(f"✅ {module_name}: {latest_file}")
        else:
            missing_files.append(f"❌ {module_name}: {pattern}")
    
    return available_files, missing_files

def main():
    print("🎯 HR Predictive Analytics Dashboard Launcher")
    print("=" * 60)
    
    # Check if FastAPI server file exists
    if not Path("main.py").exists():
        print("❌ main.py (FastAPI server) not found!")
        print("Please ensure main.py is in the same directory.")
        return
    
    # Check if dashboard HTML exists
    dashboard_files = ["web_interface.html", "dashboard.html"]
    dashboard_file = None
    for file in dashboard_files:
        if Path(file).exists():
            dashboard_file = file
            break
    
    if not dashboard_file:
        print("❌ Dashboard HTML file not found!")
        print("Please ensure web_interface.html is in the same directory.")
        return
    
    print(f"✅ FastAPI server: main.py")
    print(f"✅ Dashboard UI: {dashboard_file}")
    
    # Check data files
    print("\n📊 Checking Analysis Data Files:")
    available_files, missing_files = check_data_files()
    
    for file_info in available_files:
        print(f"  {file_info}")
    
    if missing_files:
        print("\n⚠️ Missing Data Files:")
        for file_info in missing_files:
            print(f"  {file_info}")
        print("\n💡 To generate missing data files, run:")
        print("   python attrition_modelling.py")
        print("   python Hiring_calculation.py")  
        print("   python Optimize_Simple_Calculation.py")
        print("   python business_growth.py")
        print("\n🚀 Dashboard will work with available data, or show guidance for missing modules.")
    else:
        print("\n🎉 All data files found! Dashboard will be fully functional.")
    
    print("\n" + "=" * 60)
    
    try:
        print("🚀 Starting FastAPI server...")
        print("📊 Dashboard will be available at: http://localhost:8000")
        print("📖 API documentation at: http://localhost:8000/docs")
        print("🌐 Opening dashboard in browser in 3 seconds...")
        
        # Start FastAPI server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
        
        # Wait a moment then open browser
        time.sleep(3)
        webbrowser.open("http://localhost:8000")
        
        print("\n✅ Dashboard is running!")
        print("📊 Access your HR Analytics Dashboard at: http://localhost:8000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Wait for user to stop
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        process.terminate()
        process.wait()
        print("✅ Dashboard stopped successfully!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Python and required packages are installed")
        print("2. Check if port 8000 is available")
        print("3. Ensure all files are in the same directory")

if __name__ == "__main__":
    main()