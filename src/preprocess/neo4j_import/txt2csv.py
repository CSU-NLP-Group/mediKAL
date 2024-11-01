import csv
from tqdm import tqdm

# 打开你的txt文件
with open('/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/department/pifuke_KG.txt', 'r', encoding="utf-8") as txt_file:
    # 创建csv写入器
    entities_writer = csv.writer(open('/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_csv_files/entities_pifuke.csv', 'w', encoding="utf-8", newline=''), delimiter='|')
    relations_writer = csv.writer(open('/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_csv_files/relations_pifuke.csv', 'w', encoding="utf-8", newline=''), delimiter='|')

    # 写入csv文件的头部
    entities_writer.writerow(['entity:ID','name', ':LABEL'])
    relations_writer.writerow([':START_ID', ':END_ID', ':TYPE'])

    # 用于存储已经写入的实体
    entities_dict = {}

    # 逐行读取txt文件
    for i, line in tqdm(enumerate(txt_file)):
        if i == 0:
            continue
        # 分割行
        head_entity, head_type, relation, tail_entity, tail_type = line.strip().split('\t')

        # 如果实体还没有被写入，就写入实体，并分配一个唯一的ID
        if head_entity not in entities_dict:
            entities_dict[head_entity] = len(entities_dict) + 1
            entities_writer.writerow([entities_dict[head_entity], head_entity, head_type])
        if tail_entity not in entities_dict:
            entities_dict[tail_entity] = len(entities_dict) + 1
            entities_writer.writerow([entities_dict[tail_entity], tail_entity, tail_type])

        # 写入关系
        relations_writer.writerow([entities_dict[head_entity], entities_dict[tail_entity], relation])
    
# import 命令
# ./bin/neo4j-admin import --database=pifuke --nodes=/home/myjia/envs/neo4j/neo4j-community-4.4.26/import/entities_pifuke.csv --relationships=/home/myjia/envs/neo4j/neo4j-community-4.4.26/import/relations_pifuke.csv --id-type=INTEGER --delimiter="|"