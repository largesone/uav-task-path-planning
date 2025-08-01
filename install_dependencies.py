#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖安装脚本
自动检测并安装缺失的依赖包
"""

import subprocess
import sys
import importlib
import os

def check_and_install_package(package_name, import_name=None):
    """检查并安装包"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ {package_name} 安装失败")
            return False

def main():
    """主函数"""
    print("检查和安装项目依赖...")
    print("=" * 50)
    
    # 核心依赖列表
    dependencies = [
        ("torch", "torch"),
        ("tensorboard", "tensorboard"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
        ("networkx", "networkx"),
        ("gymnasium", "gymnasium"),
        ("pydantic", "pydantic")
    ]
    
    failed_packages = []
    
    for package_name, import_name in dependencies:
        if not check_and_install_package(package_name, import_name):
            failed_packages.append(package_name)
    
    print("\n" + "=" * 50)
    if failed_packages:
        print(f"安装失败的包: {', '.join(failed_packages)}")
        print("请手动安装这些包或检查网络连接")
        return False
    else:
        print("所有依赖安装完成！")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)