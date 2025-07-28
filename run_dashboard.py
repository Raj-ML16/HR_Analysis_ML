#!/usr/bin/env python3
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
        
        print("\nDashboard is running!")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Wait for user to stop
        process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        process.terminate()
        process.wait()
        print("Dashboard stopped successfully!")
    except Exception as e:
        print(f"Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
