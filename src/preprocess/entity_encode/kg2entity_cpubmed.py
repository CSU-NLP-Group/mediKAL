from tqdm import tqdm
import json
import re
import pandas as pd

f = open("/home/myjia/Medical_LLM_task/KG/CPubMed-KGv1_1.txt", "r", encoding="utf-8")
fout_KG = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/CPubMed-KG.txt", "w", encoding="utf-8")
fout_entity2id = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_entities2id.txt", "w", encoding="utf-8")
fout_relation2id = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_relations2id.txt", "w", encoding="utf-8")
fout_type2id = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_types2id.txt", "w", encoding="utf-8")
# 首先对fout_KG第一行写入表头：head_entity, head_type, relation, tail_entity, tail_type
fout_KG.write("head_entity\thead_type\trelation\ttail_entity\ttail_type\n")
fout_KG.flush()
id_entity = 0
id_relation = 0
entity_type_id = 0
entity_type_dict = {}
entity_dict = {}
relation_dict = {}

for index, row in tqdm(enumerate(f.readlines())): 
    if index == 0:
        continue
    row = row.strip().split("\t")
    if len(row) != 3:
        print("warning: line " + str(index) + " is not a triplet!")
        print("content: " + str(row))
        
    head_name = row[0]
    relation = row[1]
    tail_name = row[2]
    head_entity = head_name.split("@@")[0]
    if len(head_name.split("@@")) == 2:
        head_type = head_name.split("@@")[1]
    else:
        head_type = "None"
    # if head_type is None:
    #     head_type = "None"
    tail_entity = tail_name.split("@@")[0]
    if len(tail_name.split("@@")) == 2:
        tail_type = tail_name.split("@@")[1]
    else:
        tail_type = "None"
    # if tail_type is None:
    #     tail_type = "None"
    if head_entity not in entity_dict:
        entity_dict[head_entity] = id_entity
        id_entity += 1
    if tail_entity not in entity_dict:
        entity_dict[tail_entity] = id_entity
        id_entity += 1
    if relation not in relation_dict:
        relation_dict[relation] = id_relation
        id_relation += 1
    if head_type not in entity_type_dict:
        entity_type_dict[head_type] = entity_type_id
        entity_type_id += 1
    if tail_type not in entity_type_dict:
        entity_type_dict[tail_type] = entity_type_id
        entity_type_id += 1
    
    fout_KG.write(head_entity + "\t" + head_type + "\t" + relation + "\t" + tail_entity + "\t" + tail_type + "\n")
    fout_KG.flush()

fout_KG.close()


# for line in tqdm(f.readlines()):
#     line = line.strip().split("\t")
#     s_entity = line[0]
#     e_entity = line[2]
#     relation = line[1]
#     if s_entity not in entity_dict:
#         entity_dict[s_entity] = id_entity
#         id_entity += 1
#     if e_entity not in entity_dict:
#         entity_dict[e_entity] = id_entity
#         id_entity += 1
#     if relation not in relation_dict:
#         relation_dict[relation] = id_relation
#         id_relation += 1
    
# 遍历实体字典，按照"实体\tid"的格式写入文件
for entity in entity_dict:
    fout_entity2id.write(entity + "\t" + str(entity_dict[entity]) + "\t" +  "\n")
    fout_entity2id.flush()
fout_entity2id.close()

# 遍历关系字典，按照"关系\tid"的格式写入文件
for relation in relation_dict:
    fout_relation2id.write(relation + "\t" + str(relation_dict[relation]) + "\n")
    fout_relation2id.flush()
fout_relation2id.close()

# 遍历实体类型字典，按照"实体类型\tid"的格式写入文件
for entity_type in entity_type_dict:
    fout_type2id.write(entity_type + "\t" + str(entity_type_dict[entity_type]) + "\n")
    fout_type2id.flush()
fout_type2id.close()


