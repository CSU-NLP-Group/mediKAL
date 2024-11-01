import json
from tqdm import tqdm
import readline
# 创建一个空字典来保存实体和它们的类型
entity_dict = {}

# 打开log_file并查看上一次处理到第几个样本
last_idx = 0
with open('/home/myjia/Medical_LLM_task/MindMap/preprocess/data_filter/log.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    if len(lines) > 0:
        last_line = lines[-1]
        last_idx = int(last_line.strip().split('\t')[0].split('：')[1])
    else:
        last_idx = 0

print("上一次处理到第{}个样本".format(last_idx))
f.close()


log_file = open('/home/myjia/Medical_LLM_task/MindMap/preprocess/data_filter/log.txt', 'a', encoding="utf-8")
fout  = open('/home/myjia/Medical_LLM_task/MindMap/preprocess/data_filter/modify.txt', 'a', encoding="utf-8")

# 打开txt文件并按行遍历
with open('/home/myjia/Medical_LLM_task/MindMap/preprocess/data_filter/filter.txt', 'r', encoding="utf-8") as f:
    for idx, line in tqdm(enumerate(f)):
        # 如果idx小于上一次处理到的idx，则跳过该行
        if idx <= last_idx:
            continue
        entity = line.strip()
        # 打印实体并提示用户进行判别
        print(f'实体: {entity}')
        user_input = input('请对该实体的类型进行判别: ')
        # 如果用户输入的是's'，则跳过当前实体
        if user_input == 's':
            continue
        # 否则，将实体和用户输入的类型添加到字典中
        else:
            fout.write(entity)
            fout.write("\n")
            fout.flush()

            log_file.write("处理进度：{}\t当前实体：{}\n".format(idx, entity))
            log_file.flush()

