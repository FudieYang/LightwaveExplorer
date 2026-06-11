#!/bin/bash
# 🔍 LWE Island CMA-ES 实时监控
# 用法: bash monitor.sh        (看一次)
#       bash monitor.sh watch   (每30秒自动刷新)

show_status() {
    clear
    echo "============================================================"
    echo "         🏝️ LWE Island CMA-ES Optimizer Monitor 🏝️         "
    echo "============================================================"
    echo "当前时间: $(date)"
    echo ""

    # 1. 运行进度 (基于 cma_eval_log.jsonl)
    jsonl_log="/home/fudie/LightwaveExplorer/Source/Python/cma_eval_log.jsonl"
    if [ -f "$jsonl_log" ]; then
        completed=$(wc -l < "$jsonl_log")
        echo "当前进度:"
        echo "  已完成仿真 (数据库记录数): $completed"
        size=$(ls -lh "$jsonl_log" | awk '{print $5}')
        mod_time=$(stat -c '%y' "$jsonl_log" | cut -d'.' -f1)
        echo "  JSONL数据库大小: $size | 最后更新: $mod_time"
    else
        echo "当前进度: 等待第一个结果写入数据库..."
    fi
    echo "────────────────────────────────────────────────────────────"

    # 2. 主控制节点 (Jupyter Kernel)
    jupyter_pid=$(ps aux | grep "[i]pykernel_launcher" | awk '{print $2}' | head -1)
    if [ -n "$jupyter_pid" ]; then
        etime=$(ps -p "$jupyter_pid" -o etime= | xargs)
        echo "主控制节点 (Jupyter Kernel): 存活 (实际挂机时长: $etime)"
    else
        echo "主控制节点: 未检测到 Jupyter Kernel"
    fi
    echo ""

    # 3. LWE 仿真进程
    lwe_procs=$(ps aux | grep "[b]uild_cli/LightwaveExplorer")
    if [ -n "$lwe_procs" ]; then
        echo "正在运行的物理引擎 (各岛屿):"
        echo "$lwe_procs" | while read line; do
            pid=$(echo "$line" | awk '{print $2}')
            etime=$(ps -p "$pid" -o etime= | xargs)
            island=$(echo "$line" | grep -oP 'lwe_tmp/\K[^/]+')
            if [ -z "$island" ]; then
                island="Unknown"
            fi
            echo "  PID: $pid | 运行时长: $etime | 节点池: $island"
        done
    else
        echo "没有底层 LWE 引擎在运行 (可能在跨岛屿迁移或梯度更新)"
    fi
    echo "────────────────────────────────────────────────────────────"

    # 4. 岛屿工作目录监控
    tmp_dir="/mnt/d/lwe_tmp"
    echo "当前临时运行池 ($tmp_dir):"
    if [ -d "$tmp_dir" ]; then
        count=$(ls $tmp_dir 2>/dev/null | wc -l)
        echo "  活跃的工作目录数量: $count"
        ls -lt $tmp_dir 2>/dev/null | head -6 | tail -n +2 | while read line; do
            name=$(echo "$line" | awk '{print $NF}')
            date_str=$(echo "$line" | awk '{print $6, $7, $8}')
            echo "  $date_str  $name"
        done
    else
        echo "  (目录未创建或为空)"
    fi
    echo "────────────────────────────────────────────────────────────"

    # 5. GPU 显卡监控
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU 显卡负载实时监控:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader | awk -F', ' '{print "  GPU "$1" ("$2"): 算力 "$3" | 温度 "$4"°C | 功耗 "$5" | 显存 "$6" / "$7}'
    else
        echo "GPU 显卡监控: 未检测到 nvidia-smi"
    fi
    echo "============================================================"
}

if [ "$1" = "watch" ]; then
    while true; do
        show_status
        echo "30秒后刷新... (Ctrl+C 退出)"
        sleep 30
    done
else
    show_status
fi
