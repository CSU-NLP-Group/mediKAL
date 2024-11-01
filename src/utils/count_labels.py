import json
import os

fin_dir = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/multi-choice_QA/CMB/CMB-Clin/"

total_label_num = 0
total_emr_num = 0

for f in os.listdir(fin_dir):
    if f.endswith(".json"):
        fin = open(fin_dir + f, 'r', encoding='utf-8')
        data = json.load(fin)
        fin.close()
        for EMR_dict in data:
            total_label_num += len(EMR_dict["label"])
            total_emr_num += 1

print(total_label_num, total_emr_num)
# 平均值
print(total_label_num / total_emr_num)

        