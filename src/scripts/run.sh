#!/bin/bash

# 切换目录
cd ../main/

# 运行 main_EMR.py
python main_EMR.py

# 检查上一个命令是否成功执行
if [ $? -eq 0 ]; then
    # 如果 main_EMR.py 成功执行，那么运行 evaluate.py
    python evaluate.py
else
    # 如果 main_EMR.py 执行失败，那么打印错误消息并退出
    echo "main_EMR.py failed, not running evaluate.py"
    exit 1
fi