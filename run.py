#!/usr/bin/env python3
"""
Riyadh Metro Assistant - Startup Script
This script helps you start the application and index data easily.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_env_file():
    """Check if .env file exists and has OpenAI API key"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ Error: .env file not found")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        if 'OPENAI_API_KEY' not in content or 'your_api_key_here' in content:
            print("❌ Error: Please set your OpenAI API key in the .env file")
            return False
    
    print("✅ Environment file configured")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        # Try uv first, fallback to pip
        try:
            subprocess.run(["uv", "sync"], check=True, capture_output=True)
            print("✅ Dependencies installed with uv")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  uv not found, trying pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed with pip")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting Riyadh Metro Assistant...")
    
    # Change to repository directory
    os.chdir("repository")
    
    try:
        # Start the server in the background
        process = subprocess.Popen([sys.executable, "main.py"])
        print("✅ Server started successfully")
        print("🌐 Application running at: http://localhost:3000")
        return process
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def wait_for_server():
    """Wait for the server to be ready"""
    print("⏳ Waiting for server to be ready...")
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get("http://localhost:3000/", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}/{max_attempts})")
    
    print("❌ Server failed to start within 30 seconds")
    return False

def index_data():
    """Index all data files"""
    print("📚 Indexing data files...")
    try:
        response = requests.post("http://localhost:3000/index_all_data", timeout=60)
        if response.status_code == 200:
            result = response.json()
            print("✅ Data indexed successfully!")
            print(f"📊 Collections created: {', '.join(result.get('collections_created', []))}")
            return True
        else:
            print(f"❌ Error indexing data: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Riyadh Metro Assistant - Startup Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    if not check_env_file():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Start server
    server_process = start_server()
    if not server_process:
        return 1
    
    try:
        # Wait for server to be ready
        if not wait_for_server():
            return 1
        
        # Index data
        if not index_data():
            print("⚠️  Data indexing failed, but server is running")
            print("   You can manually index data by visiting: http://localhost:3000")
        
        print("\n🎉 Setup complete!")
        print("📖 You can now:")
        print("   - Visit http://localhost:3000 to use the web interface")
        print("   - Use the API endpoints for programmatic access")
        print("   - Press Ctrl+C to stop the server")
        
        # Keep the server running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
