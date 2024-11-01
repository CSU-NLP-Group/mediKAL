import json
from tqdm import tqdm

# fin1 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/jingshen_pifu.txt", "r", encoding="utf-8")
# fin2 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/KG_merge.txt", "r", encoding="utf-8")

# fout1 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_entities2id_small.txt", "w", encoding="utf-8")
# fout2 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_entities2id_merge.txt", "w", encoding="utf-8")

# fout3 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/entity_type_map_small.json", "w", encoding="utf-8")
# fout4 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/entity_type_map_merge.json", "w", encoding="utf-8")
fin = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/department/pifuke_KG.txt", "r", encoding="utf-8")
fout1 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/KG_entity2id/KG_entities2id_pifuke.txt", "w", encoding="utf-8")
fout2 = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/entity_type_maps/entity_type_map_pifuke.json", "w", encoding="utf-8")
# fin1的头部是：head_entity	head_type	relation	tail_entity	tail_type，因此第一行我们不读取，从第二行开始读取；
# 每一行的第一个元素是head_entity，第二个元素是head_type，第三个元素是relation，第四个元素是tail_entity，第五个元素是tail_type，用制表符分隔
# 对于每一行，head_entity和tail_entity都是实体，需要写入fout1中，但是实体会有重复，导致id错乱，因此需要一个字典来记录实体和id的对应关系

entity2id = {}
entity_id = 0

entity_type_map = {}
for i, line in tqdm(enumerate(fin)):
    line = line.strip()
    if line == "" or i == 0:
        continue
    line_list = line.split("\t")
    head_entity = line_list[0]
    tail_entity = line_list[3]
    if head_entity not in entity2id:
        entity2id[head_entity] = entity_id
        entity_id += 1
    if tail_entity not in entity2id:
        entity2id[tail_entity] = entity_id
        entity_id += 1
    
    head_type = line_list[1]
    tail_type = line_list[4]
    if head_entity not in entity_type_map:
        entity_type_map[head_entity] = head_type
    if tail_entity not in entity_type_map:
        entity_type_map[tail_entity] = tail_type

for key, value in entity2id.items():
    fout1.write("{}\t{}\n".format(key, value))
    fout1.flush()

json.dump(entity_type_map, fout2, ensure_ascii=False, indent=4)

# entity2id = {}
# entity_id = 0

# entity_type_map = {}

# for i, line in tqdm(enumerate(fin2)):
#     line = line.strip()
#     if line == "" or i == 0:
#         continue
#     line_list = line.split("\t")
#     head_entity = line_list[0]
#     tail_entity = line_list[3]
#     if head_entity not in entity2id:
#         entity2id[head_entity] = entity_id
#         entity_id += 1
#     if tail_entity not in entity2id:
#         entity2id[tail_entity] = entity_id
#         entity_id += 1
    
#     head_type = line_list[1]
#     tail_type = line_list[4]
#     if head_entity not in entity_type_map:
#         entity_type_map[head_entity] = head_type
#     if tail_entity not in entity_type_map:
#         entity_type_map[tail_entity] = tail_type

# for key, value in entity2id.items():
#     fout2.write("{}\t{}\n".format(key, value))
#     fout2.flush()

# json.dump(entity_type_map, fout4, ensure_ascii=False, indent=4)