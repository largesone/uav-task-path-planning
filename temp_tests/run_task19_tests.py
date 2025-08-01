#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务19测试执行脚本
一键运行所有端到端系统集成测试
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

def run_test_script(script_path: str, test_name: str) -> dict:
    """运行单个测试脚本"""
    print(f"\n{'='*60}")
    print(f"执行测试: {test_name}")
    print(f"脚本路径: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 执行测试脚本
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        duration = time.time() - start_time
        
        # 解析结果
        test_result = {
            "test_name": test_name,
            "script_path": script_path,
            "duration": duration,
            "return_code": result.returncode,
            "status": "PASSED" if result.returncode == 0 else "FAILED",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        print(f"测试完成: {test_result['status']} (耗时: {duration:.2f}秒)")
        
        if result.returncode != 0:
            print(f"错误输出:\n{result.stderr}")
        
        return test_result
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"测试超时: {test_name} (耗时: {duration:.2f}秒)")
        return {
            "test_name": test_name,
            "script_path": script_path,
            "duration": duration,
            "return_code": -1,
            "status": "TIMEOUT",
            "stdout": "",
            "stderr": "Test execution timeout"
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"测试执行异常: {test_name} - {e}")
        return {
            "test_name": test_name,
            "script_path": script_path,
            "duration": duration,
            "return_code": -2,
            "status": "ERROR",
            "stdout": "",
            "stderr": str(e)
        }

def main():
    """主函数"""
    print("任务19: 端到端系统集成测试执行器")
    print("=" * 80)
    
    # 获取当前目录
    current_dir = Path(__file__).parent
    
    # 定义测试脚本
    test_scripts = [
        {
            "name": "端到端集成测试",
            "script": "test_end_to_end_integration.py",
            "description": "完整的系统集成测试，包括课程学习、零样本迁移、指标验证等"
        },
        {
            "name": "分布式训练稳定性测试", 
            "script": "test_distributed_training_stability.py",
            "description": "Ray RLlib分布式训练的稳定性和性能测试"
        }
    ]
    
    # 执行所有测试
    all_results = []
    total_start_time = time.time()
    
    for test_config in test_scripts:
        script_path = current_dir / test_config["script"]
        
        if not script_path.exists():
            print(f"⚠ 测试脚本不存在: {script_path}")
            continue
        
        print(f"\n📋 准备执行: {test_config['name']}")
        print(f"📝 描述: {test_config['description']}")
        
        result = run_test_script(str(script_path), test_config["name"])
        all_results.append(result)
    
    # 生成总体报告
    total_duration = time.time() - total_start_time
    
    passed_tests = sum(1 for r in all_results if r["status"] == "PASSED")
    failed_tests = sum(1 for r in all_results if r["status"] == "FAILED")
    timeout_tests = sum(1 for r in all_results if r["status"] == "TIMEOUT")
    error_tests = sum(1 for r in all_results if r["status"] == "ERROR")
    total_tests = len(all_results)
    
    # 保存详细结果
    report = {
        "task_id": 19,
        "task_name": "端到端系统集成测试",
        "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration": total_duration,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "timeout_tests": timeout_tests,
            "error_tests": error_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        },
        "test_results": all_results
    }
    
    # 保存报告
    report_file = current_dir / "task19_execution_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("任务19测试执行总结")
    print(f"{'='*80}")
    print(f"📊 总测试数量: {total_tests}")
    print(f"✅ 通过测试: {passed_tests}")
    print(f"❌ 失败测试: {failed_tests}")
    print(f"⏰ 超时测试: {timeout_tests}")
    print(f"💥 错误测试: {error_tests}")
    print(f"📈 成功率: {report['summary']['success_rate']:.1%}")
    print(f"⏱ 总耗时: {total_duration:.2f}秒")
    print(f"📄 详细报告: {report_file}")
    
    # 详细结果
    print(f"\n详细测试结果:")
    for result in all_results:
        status_emoji = {
            "PASSED": "✅",
            "FAILED": "❌", 
            "TIMEOUT": "⏰",
            "ERROR": "💥"
        }
        emoji = status_emoji.get(result["status"], "❓")
        print(f"  {emoji} {result['test_name']}: {result['status']} ({result['duration']:.2f}s)")
    
    # 设置退出码
    if failed_tests == 0 and timeout_tests == 0 and error_tests == 0:
        print(f"\n🎉 所有测试通过！任务19完成！")
        return 0
    else:
        print(f"\n⚠ 存在测试问题，请检查详细报告")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)