import pandas as pd
from tqdm import tqdm


df = pd.read_excel("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/original_kg_excel_files/pifu_jingshen.xlsx", sheet_name=0)

fout = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_txt_files/jingshen_new.txt", "w", encoding="utf-8")

relation_map = {"must_see_in": "必见于", "almost_see_in": "典型见于", "main_see_in": "主要见于", "may_see_in": "一般见于", "less_see_in": "少见于", "must_result_to": "必然表现",
                "almost_result_to": "典型表现", "main_result_to": "主要表现", "may_result_to": "一般表现", "less_result_to":"罕见表现", "not_result_to": "否定表现"}

fout.write("{}\t{}\t{}\t{}\t{}\n".format("head_entity", "head_type", "relation", "tail_entity", "tail_type"))

for index, row in tqdm(df.iterrows(), total = df.shape[0]):
    head_entity = row['head_entity']
    relation = row['relation']
    relation = relation_map[relation]
    tail_entity = row['tail_entity']

    if "见于" in relation:
        head_type = "症状"
        tail_type = "疾病"
    else:
        head_type = "疾病"
        tail_type = "症状"

    fout.write("{}\t{}\t{}\t{}\t{}\n".format(head_entity, head_type, relation, tail_entity, tail_type))
    fout.flush()


    