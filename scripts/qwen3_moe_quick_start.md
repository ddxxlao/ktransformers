# Qwen3 MoE 性能监控快速开始指南

## 概述

本指南将帮助您快速设置和运行Qwen3 MoE模型的性能监控测试，特别是验证AMX优化的效果。

## 前置条件

### 硬件要求
- **CPU**: Intel Xeon 4th Gen (支持AMX) 或更新版本
- **内存**: 至少64GB，推荐512GB+
- **GPU**: RTX 3090/4090 24GB 或更高
- **存储**: NVMe SSD，至少500GB可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04+ 或 CentOS 8+
- **Python**: 3.11
- **CUDA**: 12.1+
- **Docker**: 可选，用于容器化部署

## 快速安装

### 1. 环境准备

```bash
# 检查AMX支持
lscpu | grep -i amx
# 应该看到: amx-bf16 amx-int8 amx-tile

# 创建工作目录
mkdir -p ~/qwen3_moe_test
cd ~/qwen3_moe_test

# 创建Python虚拟环境
conda create -n qwen3_moe python=3.11
conda activate qwen3_moe
```

### 2. 安装KTransformers

```bash
# 克隆仓库
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install packaging ninja cpufeature numpy aiohttp pyyaml

# 编译安装（双路CPU）
export USE_NUMA=1
bash install.sh

# 或单路CPU
# bash install.sh
```

### 3. 下载模型

```bash
# 创建模型目录
mkdir -p ~/models/qwen3moe

# 下载Qwen3 MoE 30B-A3B模型（示例）
# 方法1: 使用huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-MoE-30B-A3B --local-dir ~/models/qwen3moe

# 方法2: 使用git lfs
cd ~/models
git lfs clone https://huggingface.co/Qwen/Qwen3-MoE-30B-A3B qwen3moe

# 下载GGUF格式权重
mkdir -p ~/models/qwen3moe-gguf
# 从HuggingFace下载对应的GGUF文件到此目录
```

## 快速测试

### 1. 启动服务器
有可能卡到flashiner编译，这时候需要检查我们的 `export TORCH_CUDA_ARCH_LIST="8.9"`，最好只选择自己的GPU的架构。
`python -c "import torch; print(torch.cuda.get_device_capability())"` 来查看架构信息。

#### 标准后端（非AMX）
```bash
cd ~/qwen3_moe_test/ktransformers

# GPU 预填 + CPU 解码（默认）
export TORCH_CUDA_ARCH_LIST="8.9"
python ktransformers/server/main.py \
  --architectures Qwen3MoeForCausalLM \
  --model_path /workspace/data/models/qwen3moe \
  --gguf_path /workspace/data/models/qwen3moe-gguf/2507/q8 \
  --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve.yaml \
  --backend_type balance_serve
```

> 提示：`Qwen3Moe-serve.yaml` 会自动把 Prefill 计算放在 GPU、推理阶段专家放在 CPU；同时仍然需要 GGUF 权重。若想全程都在 GPU，可在 YAML 中把 `prefill_device`、`generate_device` 均改成 `"cuda"` 并把 `generate_op` 调整为 `"KExpertsTorch"`，或直接使用 `--backend_type ktransformers` 走单进程推理（牺牲并发调度换取部署简单）。

#### AMX 优化后端
```bash
cd ~/qwen3_moe_test/ktransformers

export TORCH_CUDA_ARCH_LIST="8.9"
python ktransformers/server/main.py \
  --architectures Qwen3MoeForCausalLM \
  --model_path /workspace/data/models/qwen3moe \
  --gguf_path /workspace/data/models/qwen3moe-gguf/BF16 \
  --optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve-amx.yaml \
  --backend_type balance_serve
```

### 2. 验证服务器状态

```bash
# 新开终端，测试服务器是否正常
curl -X POST http://localhost:10002/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "model": "Qwen3MoE",
    "max_tokens": 50,
    "return_speed": true
  }'
```

### 3. 运行性能测试

```bash
# 基础性能测试
python ./qwen3_moe_performance_test.py \
  --api_url http://localhost:10002/v1/chat/completions \
  --model Qwen3MoE \
  --prompt_lengths 1024 2048 4096 \
  --max_tokens 300 \
  --num_runs 3 \
  --output_dir ./performance_results

# 查看测试结果
ls -la ./performance_results/
cat ./performance_results/qwen3_moe_performance_report_*.txt
```

## AMX优化效果验证

### 1. 对比测试

```bash
# 运行AMX vs 标准配置对比测试
python ktransformers/tests/amx_comparison_test.py \
  --model_path ~/models/qwen3moe \
  --gguf_path ~/models/qwen3moe-gguf \
  --cpu_cores 62 \
  --prompt_lengths 1024 2048 4096 \
  --num_runs 3 \
  --output_dir ./amx_comparison_results

# 查看对比结果
cat ./amx_comparison_results/amx_comparison_report_*.txt
```

### 2. 预期结果

正常情况下，您应该看到类似以下的性能提升：

```
性能对比汇总:
Prompt长度    配置      Prefill(t/s)    Decode(t/s)     首Token(s)    提升倍数
1024         标准      85.32           13.45           0.852         1.00x
1024         AMX       185.67          13.52           0.652         2.18x

2048         标准      98.45           13.28           1.324         1.00x
2048         AMX       234.67          13.35           1.124         2.38x

4096         标准      112.34          13.15           2.489         1.00x
4096         AMX       278.91          13.22           2.089         2.48x
```

## 故障排除

### 常见问题

#### 1. AMX不支持
```bash
# 检查CPU特性
lscpu | grep -E "(amx|avx)"

# 如果没有AMX支持，使用标准配置
--optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve.yaml
```

#### 2. 内存不足
```bash
# 检查内存使用
free -h

# 减少批处理大小
--max_batch_size 2
--chunk_size 4096
```

#### 3. CUDA错误
```bash
# 检查CUDA状态
nvidia-smi

# 检查CUDA版本兼容性
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. 服务器启动失败
```bash
# 检查端口占用
netstat -tlnp | grep 10002

# 查看详细错误日志
python ktransformers/server/main.py ... --verbose
```

### 性能调优

#### 1. CPU核心数优化
```bash
# 查看CPU核心数
nproc

# 双路CPU (64核)
--cpu_infer 62

# 单路CPU (32核)
--cpu_infer 30
```

#### 2. 内存优化
```bash
# 启用大页内存
echo always > /sys/kernel/mm/transparent_hugepages/enabled

# NUMA优化（双路CPU）
numactl --cpunodebind=0,1 --membind=0,1 python ktransformers/server/main.py ...
```

#### 3. 网络优化
```bash
# 增加网络缓冲区
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p
```

## 监控和维护

### 1. 实时监控

```bash
# 监控CPU使用率
htop -d 1

# 监控内存使用
watch -n 1 'free -h'

# 监控GPU使用率
nvidia-smi -l 1

# 监控网络连接
watch -n 1 'netstat -an | grep 10002'
```

### 2. 日志分析

```bash
# 查看服务器日志
tail -f /var/log/ktransformers/server.log

# 分析性能日志
grep "Performance" /var/log/ktransformers/server.log | tail -20

# 查看错误日志
grep "ERROR" /var/log/ktransformers/server.log
```

### 3. 定期测试

```bash
# 创建定期测试脚本
cat > ~/qwen3_moe_test/daily_test.sh << 'EOF'
#!/bin/bash
cd ~/qwen3_moe_test/ktransformers

# 运行每日性能测试
python ktransformers/tests/qwen3_moe_performance_test.py \
  --api_url http://localhost:10002/v1/chat/completions \
  --model Qwen3MoE \
  --prompt_lengths 2048 \
  --max_tokens 200 \
  --num_runs 1 \
  --output_dir ./daily_results/$(date +%Y%m%d)

# 发送结果通知（可选）
# python send_performance_report.py
EOF

chmod +x ~/qwen3_moe_test/daily_test.sh

# 添加到crontab（每天上午9点运行）
echo "0 9 * * * ~/qwen3_moe_test/daily_test.sh" | crontab -
```

## 进阶配置

### 1. 多GPU配置

```bash
# 使用多GPU配置文件
--optimize_config_path ktransformers/optimize/optimize_rules/Qwen3Moe-serve-multi-gpu.yaml

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0,1
```

### 2. 容器化部署

```bash
# 构建Docker镜像
docker build -t qwen3-moe-ktransformers .

# 运行容器
docker run -d \
  --name qwen3-moe \
  --gpus all \
  -p 10002:10002 \
  -v ~/models:/models \
  qwen3-moe-ktransformers
```

### 3. 负载均衡

```bash
# 使用nginx进行负载均衡
cat > /etc/nginx/conf.d/qwen3-moe.conf << 'EOF'
upstream qwen3_moe {
    server 127.0.0.1:10002;
    server 127.0.0.1:10003;
    server 127.0.0.1:10004;
}

server {
    listen 80;
    location /v1/chat/completions {
        proxy_pass http://qwen3_moe;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

systemctl reload nginx
```

## 总结

通过本快速开始指南，您应该能够：

1. ✅ 成功安装和配置KTransformers
2. ✅ 启动Qwen3 MoE模型服务
3. ✅ 运行性能测试并验证AMX优化效果
4. ✅ 监控和维护系统性能
5. ✅ 解决常见问题和进行性能调优

如果遇到问题，请参考：
- [详细性能监控分析文档](./performance_monitoring_analysis.md)
- [官方FAQ](../en/FAQ.md)
- [GitHub Issues](https://github.com/kvcache-ai/ktransformers/issues)

---

*祝您使用愉快！如有问题欢迎反馈。*
