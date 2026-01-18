#!/usr/bin/env python
"""
A股 Dashboard 启动脚本

使用方式:
    python run_astock_dashboard.py
    
或者直接:
    streamlit run astock_dashboard/app.py
"""
import subprocess
import sys
import os

def main():
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(root_dir, "astock_dashboard", "app.py")
    
    # 检查文件是否存在
    if not os.path.exists(app_path):
        print(f"错误: 找不到 {app_path}")
        sys.exit(1)
    
    print("=" * 50)
    print("A股 AlphaGPT 监控面板")
    print("=" * 50)
    print(f"启动路径: {app_path}")
    print("访问地址: http://localhost:8501")
    print("-" * 50)
    
    # 启动 Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])

if __name__ == "__main__":
    main()

