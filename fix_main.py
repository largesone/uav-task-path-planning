#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 修复main.py文件的脚本
import re

def fix_main_file():
    """修复main.py文件中的语法错误"""
    
    # 读取文件
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找并移除错误的代码行
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是错误的缩进行
        if '扁平模式 - 状态向量长度' in line and line.startswith('        '):
            print(f"发现错误行 {i+1}: {line.strip()}")
            # 跳过这行和可能的相关错误行
            i += 1
            continue
        
        # 检查其他可能的错误模式
        if 'input_dim' in line and line.startswith('        print(f"'):
            print(f"发现可能的错误行 {i+1}: {line.strip()}")
            i += 1
            continue
            
        # 检查是否有孤立的else语句
        if line.strip() == 'else:  # graph mode' and i > 0:
            prev_line = lines[i-1].strip()
            if not prev_line.endswith(':') and 'if' not in prev_line:
                print(f"发现孤立的else语句 {i+1}: {line.strip()}")
                # 跳过这个else块
                i += 1
                while i < len(lines) and (lines[i].startswith('        ') or lines[i].strip() == ''):
                    i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    # 写回文件
    fixed_content = '\n'.join(fixed_lines)
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("文件修复完成")

if __name__ == "__main__":
    fix_main_file()