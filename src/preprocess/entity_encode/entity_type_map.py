from tqdm import tqdm
import json

fin = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/entity_type_map.json", "r", encoding="utf-8")

entity_type_map = json.load(fin)
print(len(entity_type_map))

# fin = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/CPubMed-KG.txt", "r", encoding="utf-8")

# fout = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/entities_and_keywords/entities/entity_type_map.json", "w", encoding="utf-8")

# entity_type_map = {}
# for index, row in tqdm(enumerate(fin.readlines())):
#     if index == 0:
#         continue
#     row = row.strip().split("\t")
#     head_name = row[0]
#     head_type = row[1]
#     tail_name = row[3]
#     tail_type = row[4]
#     if head_name not in entity_type_map:
#         entity_type_map[head_name] = head_type
#     if tail_name not in entity_type_map:
#         entity_type_map[tail_name] = tail_type

# json.dump(entity_type_map, fout, ensure_ascii=False, indent=4)


# fin.close()
# fout.close()
    