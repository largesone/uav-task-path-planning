#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 简单的语法检查脚本
import ast
import sys

def check_syntax(filename):
    """检查Python文件的语法"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 尝试解析AST
        ast.parse(source)
        print(f"✅ {filename} 语法检查通过")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filename} 语法错误:")
        print(f"  行号: {e.lineno}")
        print(f"  列号: {e.offset}")
        print(f"  错误: {e.msg}")
        print(f"  文本: {e.text.strip() if e.text else 'N/A'}")
        return False
        
    except Exception as e:
        print(f"❌ {filename} 检查失败: {e}")
        return False

if __name__ == "__main__":
    if check_syntax("main.py"):
        print("可以尝试导入模块...")
        try:
            import main
            print("✅ 模块导入成功")
        except Exception as e:
            print(f"❌ 模块导入失败: {e}")
    else:
        print("❌ 语法检查失败，无法导入模块")