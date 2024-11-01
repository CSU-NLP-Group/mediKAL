import json
import os

fin_directory = "/home/myjia/Medical_LLM_task/EMR_diagnos/data/EMR_all_from_iiyi/processed_json_files/zh/"

overall_tokens = 0
overall_emr = 0

for file_name in os.listdir(fin_directory):
    if file_name.endswith('.json'):
        with open(fin_directory + file_name, 'r', encoding="utf-8") as f:
            data = json.load(f)
            total_tokens = 0
            total_emr = 0
            for line in data:
                total_emr += 1
                overall_emr += 1
                if "主诉" in line:
                    total_tokens += len(line["主诉"])
                    overall_tokens += len(line["主诉"])
                if "现病史" in line:
                    total_tokens += len(line["现病史"])
                    overall_tokens += len(line["现病史"])
                if "既往史" in line:
                    total_tokens += len(line["既往史"])
                    overall_tokens += len(line["既往史"])
                if "查体" in line:
                    total_tokens += len(line["查体"])
                    overall_tokens += len(line["查体"])
                if "辅助检查" in line:
                    total_tokens += len(line["辅助检查"])
                    overall_tokens += len(line["辅助检查"])
            
            print(f"科室：{file_name.split('.')[0]}\t病例数：{total_emr}\t总字数：{total_tokens}\t平均字数：{total_tokens/total_emr}")
        
        f.close()

print(f"总病例数：{overall_emr}\t总字数：{overall_tokens}\t平均字数：{overall_tokens/overall_emr}")
        