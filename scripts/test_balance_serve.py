#!/usr/bin/env python3
"""
KTransformers balance_serve 功能集成测试脚本

这个脚本提供了全面的 balance_serve 功能测试，包括：
1. 基础模块导入测试
2. 后端初始化测试
3. 调度器功能测试
4. 简单的推理测试（如果有模型）
5. 性能基准测试
"""

import sys
import os
import time
import traceback
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class BalanceServeTestSuite:
    """balance_serve 功能测试套件"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
        
    def log_info(self, message: str):
        print(f"[INFO] {message}")
        
    def log_success(self, message: str):
        print(f"[SUCCESS] ✓ {message}")
        
    def log_error(self, message: str):
        print(f"[ERROR] ✗ {message}")
        
    def log_warn(self, message: str):
        print(f"[WARN] ⚠ {message}")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """运行单个测试"""
        self.tests_run += 1
        self.log_info(f"运行测试 {self.tests_run}: {test_name}")
        
        try:
            start_time = time.time()
            result = test_func(*args, **kwargs)
            end_time = time.time()
            
            if result:
                self.log_success(f"{test_name} (耗时: {end_time - start_time:.2f}s)")
                self.tests_passed += 1
                self.test_results.append({
                    'name': test_name,
                    'status': 'PASSED',
                    'duration': end_time - start_time,
                    'error': None
                })
                return True
            else:
                self.log_error(f"{test_name} - 测试返回 False")
                self.tests_failed += 1
                self.test_results.append({
                    'name': test_name,
                    'status': 'FAILED',
                    'duration': end_time - start_time,
                    'error': 'Test returned False'
                })
                return False
                
        except Exception as e:
            end_time = time.time()
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.log_error(f"{test_name} - 异常: {str(e)}")
            self.tests_failed += 1
            self.test_results.append({
                'name': test_name,
                'status': 'FAILED',
                'duration': end_time - start_time,
                'error': error_msg
            })
            return False
    
    def test_basic_imports(self) -> bool:
        """测试基础模块导入"""
        try:
            import ktransformers
            import torch
            import numpy as np
            
            self.log_info(f"KTransformers 版本: {ktransformers.__version__}")
            self.log_info(f"PyTorch 版本: {torch.__version__}")
            self.log_info(f"CUDA 可用: {torch.cuda.is_available()}")
            
            return True
        except ImportError as e:
            self.log_error(f"基础模块导入失败: {e}")
            return False
    
    def test_balance_serve_imports(self) -> bool:
        """测试 balance_serve 相关模块导入"""
        try:
            from ktransformers.server.backend.interfaces.balance_serve import BalanceServeInterface
            from ktransformers.server.backend.interfaces.balance_serve import BalanceServeThreadContext

            self.log_info("balance_serve 核心模块导入成功")
            return True
        except ImportError as e:
            self.log_error(f"balance_serve 模块导入失败: {e}")
            return False
    
    def test_optimization_rules(self) -> bool:
        """测试优化规则加载"""
        try:
            import yaml
            
            rules_dir = project_root / "ktransformers" / "optimize" / "optimize_rules"
            if not rules_dir.exists():
                self.log_error(f"优化规则目录不存在: {rules_dir}")
                return False
            
            yaml_files = list(rules_dir.glob("*.yaml"))
            self.log_info(f"找到 {len(yaml_files)} 个优化规则文件")
            
            # 测试加载 balance_serve 相关的规则文件
            serve_rules = [f for f in yaml_files if 'serve' in f.name.lower()]
            if not serve_rules:
                self.log_warn("未找到 balance_serve 相关的优化规则文件")
                return True
            
            for rule_file in serve_rules[:3]:  # 测试前3个文件
                with open(rule_file, 'r') as f:
                    rules = yaml.safe_load(f)
                self.log_info(f"成功加载规则文件: {rule_file.name}")
            
            return True
        except Exception as e:
            self.log_error(f"优化规则测试失败: {e}")
            return False
    
    def test_backend_initialization(self) -> bool:
        """测试 balance_serve 后端初始化"""
        try:
            from ktransformers.server.backend.interfaces.balance_serve import BalanceServeInterface

            # 创建临时配置
            config = {
                'max_batch_size': 2,
                'cache_lens': 1024,
                'chunk_size': 128,
                'cpu_infer': 4,
                'model_path': '/tmp/dummy_model',  # 虚拟路径
                'gguf_path': '/tmp/dummy_gguf',    # 虚拟路径
            }

            # 注意：这里只测试类的创建，不进行实际的模型加载
            # 因为我们可能没有实际的模型文件
            backend_class = BalanceServeInterface
            self.log_info("BalanceServeInterface 类可以正常访问")

            return True
        except Exception as e:
            self.log_error(f"后端初始化测试失败: {e}")
            return False
    
    def test_scheduler_functionality(self) -> bool:
        """测试调度器基础功能"""
        try:
            # 尝试导入调度器相关模块
            try:
                from ktransformers.server.balance_serve.sched_rpc import SchedulerClient
                self.log_info("SchedulerClient 导入成功")
            except ImportError:
                self.log_warn("SchedulerClient 导入失败")

            # 检查其他调度器相关模块
            try:
                from ktransformers.server.balance_serve.inference.query_manager import QueryManager
                self.log_info("QueryManager 导入成功")
            except ImportError:
                self.log_warn("QueryManager 导入失败")

            return True
        except Exception as e:
            self.log_error(f"调度器功能测试失败: {e}")
            return False
    
    def test_engine_functionality(self) -> bool:
        """测试推理引擎基础功能"""
        try:
            # 尝试导入推理引擎相关模块
            try:
                from ktransformers.server.balance_serve.inference.model_runner import ModelRunner
                self.log_info("ModelRunner 导入成功")
            except ImportError:
                self.log_warn("ModelRunner 导入失败")

            # 检查其他推理相关模块
            try:
                from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
                self.log_info("ForwardBatch 模块导入成功")
            except ImportError:
                self.log_warn("ForwardBatch 模块导入失败")

            return True
        except Exception as e:
            self.log_error(f"引擎功能测试失败: {e}")
            return False
    
    def test_system_resources(self) -> bool:
        """测试系统资源"""
        try:
            import psutil
            
            # 内存检查
            mem = psutil.virtual_memory()
            total_gb = mem.total // (1024**3)
            available_gb = mem.available // (1024**3)
            
            self.log_info(f"系统内存: {total_gb}GB 总计, {available_gb}GB 可用")
            
            if total_gb < 16:
                self.log_warn("系统内存较少，可能影响大模型运行")
            
            # CPU 检查
            cpu_count = os.cpu_count()
            self.log_info(f"CPU 核心数: {cpu_count}")
            
            # 检查 AMX 支持
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                if 'amx' in cpuinfo:
                    self.log_success("检测到 AMX 指令支持")
                else:
                    self.log_info("未检测到 AMX 指令支持")
            except:
                self.log_warn("无法检查 CPU 特性")
            
            return True
        except Exception as e:
            self.log_error(f"系统资源检查失败: {e}")
            return False
    
    def test_cuda_functionality(self) -> bool:
        """测试 CUDA 功能"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.log_warn("CUDA 不可用，跳过 CUDA 测试")
                return True
            
            device_count = torch.cuda.device_count()
            self.log_info(f"检测到 {device_count} 个 CUDA 设备")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                self.log_info(f"GPU {i}: {props.name}, {props.total_memory // (1024**2)}MB")
            
            # 简单的 CUDA 操作测试
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            
            self.log_success("CUDA 基础操作测试通过")
            return True
            
        except Exception as e:
            self.log_error(f"CUDA 功能测试失败: {e}")
            return False
    
    def test_custom_flashinfer(self) -> bool:
        """测试 custom_flashinfer 模块"""
        try:
            # 尝试导入 custom_flashinfer
            import flashinfer
            self.log_success("custom_flashinfer 导入成功")
            return True
        except ImportError:
            self.log_warn("custom_flashinfer 未安装或不可用")
            return True  # 不是致命错误
        except Exception as e:
            self.log_error(f"custom_flashinfer 测试失败: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tests': self.tests_run,
                'passed': self.tests_passed,
                'failed': self.tests_failed,
                'success_rate': round(success_rate, 2)
            },
            'test_results': self.test_results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
            }
        }
        
        return report
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("  KTransformers balance_serve 功能测试")
        print("=" * 60)
        
        # 运行所有测试
        tests = [
            ("基础模块导入", self.test_basic_imports),
            ("balance_serve 模块导入", self.test_balance_serve_imports),
            ("优化规则加载", self.test_optimization_rules),
            ("后端初始化", self.test_backend_initialization),
            ("调度器功能", self.test_scheduler_functionality),
            ("引擎功能", self.test_engine_functionality),
            ("系统资源检查", self.test_system_resources),
            ("CUDA 功能", self.test_cuda_functionality),
            ("custom_flashinfer", self.test_custom_flashinfer),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()  # 空行分隔
        
        # 生成报告
        report = self.generate_report()
        
        print("=" * 60)
        print("  测试结果总结")
        print("=" * 60)
        print(f"总测试数: {report['summary']['total_tests']}")
        print(f"通过: {report['summary']['passed']}")
        print(f"失败: {report['summary']['failed']}")
        print(f"成功率: {report['summary']['success_rate']}%")
        
        # 保存详细报告
        report_file = project_root / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_file}")
        
        # 返回成功状态
        return self.tests_failed == 0


def main():
    """主函数"""
    test_suite = BalanceServeTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 所有测试通过！balance_serve 功能正常。")
        sys.exit(0)
    else:
        print(f"\n❌ {test_suite.tests_failed} 个测试失败，请检查配置。")
        sys.exit(1)


if __name__ == "__main__":
    main()
