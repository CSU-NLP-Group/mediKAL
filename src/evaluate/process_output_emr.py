import json
import re
from tqdm import tqdm
import os

def extract_numbers_between(string, a, b):
    pattern = rf'{a}(.*?){b}'  # 通过 f-string 将参数 a 和 b 嵌入正则表达式
    match = re.search(pattern, string)
    if match:
        substring = match.group(1)  # 获取匹配到的 a 和 b 之间的子串
        numbers = re.findall(r'\d+', substring)  # 提取子串中的所有数字
        return numbers
    else:
        return []

# 参数：比例
portion = 0.6
topn = 5
score_list = ["10", "0", "1","2", "3", "4", "5", "6", "7", "8", "9"]
model = "qwen-7b-chat"

fin_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/output/2ndinter/{model}/"

# 遍历文件夹的每个json文件

for f in os.listdir(fin_dir):
    if f.endswith('_merged.json'):
        print(f)
        
        fin = open(fin_dir + f, 'r', encoding='utf-8')

        data = []

        for line in fin:
            data.append(json.loads(line))

        fin.close()

        processed_data = []

        for i, EMR_dict in tqdm(enumerate(data)):
            if f.split("_")[0] == "急诊科" and EMR_dict["index"] == 41: 
                flag = 1
            assert EMR_dict["index"] == i + 1
            EMR_dict["pred_labels"] = []

            score_threshold = 2 * (10 * portion)

            drug_list = EMR_dict["pred_response"][-1][-1]
            if len(drug_list) > 0:
                score_threshold += (10 * portion)
            
            exam_dis = EMR_dict["pred_response"][-1][-2]
            
            past_dis = EMR_dict["pred_response"][-1][-3]
            if len(past_dis) > 0:
                score_threshold += (10 * portion)

            LLM_diag = EMR_dict["pred_response"][-1][-4]
            for pred in EMR_dict["pred_response"][:len(EMR_dict["pred_response"])-1]:
                cnt = 0

                cur_dis = pred[0]
                cur_analysis = pred[1]

                total_score = 0

                cc_score = extract_numbers_between(cur_analysis, "与患者主诉吻合度得分", "\n")
                total_score += int(cc_score[0]) if len(cc_score) > 0 else 0
                
                ph_score = extract_numbers_between(cur_analysis, "与患者既往病史关联程度得分", "\n")
                if len(past_dis) > 0:
                    total_score += int(ph_score[0]) if len(ph_score) > 0 else 0
                
                dg_score = extract_numbers_between(cur_analysis, "与患者既往药物使用关联程度得分", "\n")
                if len(drug_list) > 0:
                    total_score += int(dg_score[0]) if len(dg_score) > 0 else 0
                
                ae_score = extract_numbers_between(cur_analysis, "与患者检查结果关联程度得分", "\n")
                total_score += int(ae_score[0]) if len(ae_score) > 0 else 0

                # 如果总得分高于阈值，+1
                if total_score >= score_threshold:
                    cnt += 1

                # 如果当前预测结果在模型预测列表里，或者在患者既往病史里，或者在患者检查结果里，+1
                if cur_dis in LLM_diag or cur_dis in past_dis or cur_dis in exam_dis:
                    cnt += 1

                # 如果模型最终做出的判断不是"否", +1
                final_judge = re.findall(r"6.(.+)$", cur_analysis, re.DOTALL)

                if len(final_judge) > 0 and "该疾病是否能作为诊断结果" in final_judge[0]:
                    if "否" in final_judge[0].split("该疾病是否能作为诊断结果")[1] or "不能" in final_judge[0].split("该疾病是否能作为诊断结果")[1]:
                        cnt += 0

                elif len(final_judge) > 0 and "不能作为" in final_judge[0]:
                    cnt += 0
                else:
                    cnt += 1

                if cnt >= 2:
                    EMR_dict["pred_labels"].append(cur_dis)

            if len(EMR_dict["pred_labels"]) < topn:
                remains = topn - len(EMR_dict["pred_labels"])
                # 如果总数不够topn个，缺的那部分从LLM_diag里面补
                for dis in LLM_diag:
                    if dis not in EMR_dict["pred_labels"]:
                        EMR_dict["pred_labels"].append(dis)
                        remains -= 1
                    if remains == 0:
                        break

            processed_data.append(EMR_dict)

        fout_dir = f"/home/myjia/Medical_LLM_task/EMR_diagnos/output/result/{model}/"
        if not os.path.exists(fout_dir):
            os.makedirs(fout_dir)

        fout = open(fout_dir + f.split("_")[0] + "_res.json", 'w', encoding='utf-8')

        for EMR_dict in processed_data:
            fout.write(json.dumps(EMR_dict, ensure_ascii=False) + "\n")





