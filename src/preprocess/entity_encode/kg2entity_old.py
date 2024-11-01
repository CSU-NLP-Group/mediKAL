from tqdm import tqdm
import json
import re
import pandas as pd

f = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/pifuke_KG.txt", "r", encoding="utf-8")
#
fout_entity2id = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_entities2id_pifuke.txt", "w", encoding="utf-8")
fout_relation2id = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_relations2id_pifuke.txt", "w", encoding="utf-8")

id_entity = 0
id_relation = 0

entity_dict = {}
relation_dict = {}

for line in tqdm(f.readlines()):
    line = line.strip().split("\t")
    s_entity = line[0]
    e_entity = line[2]
    relation = line[1]
    if s_entity not in entity_dict:
        entity_dict[s_entity] = id_entity
        id_entity += 1
    if e_entity not in entity_dict:
        entity_dict[e_entity] = id_entity
        id_entity += 1
    if relation not in relation_dict:
        relation_dict[relation] = id_relation
        id_relation += 1
    
# 遍历实体字典，按照"实体\tid"的格式写入文件
for entity in entity_dict:
    fout_entity2id.write(entity + "\t" + str(entity_dict[entity]) + "\n")
    fout_entity2id.flush()
fout_entity2id.close()

# 遍历关系字典，按照"关系\tid"的格式写入文件
for relation in relation_dict:
    fout_relation2id.write(relation + "\t" + str(relation_dict[relation]) + "\n")
    fout_relation2id.flush()
fout_relation2id.close()



