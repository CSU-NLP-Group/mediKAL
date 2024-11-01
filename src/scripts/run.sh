#!/bin/bash

# 切换目录
cd ../main/

# 运行 main_EMR.py
python main.py

# 检查上一个命令是否成功执行
if [ $? -eq 0 ]; then
    # 如果 main_EMR.py 成功执行，那么运行 evaluate.py
    cd ../evaluate/
    python process_output_emr.py
    python evaluate_emr.py
else
    # 如果 main_EMR.py 执行失败，那么打印错误消息并退出
    echo "main.py failed, not running evaluate.py"
    exit 1
fi