from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import math

df_database = pd.read_excel("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/zhongliuke_KG.xlsx")
f = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/zhongliuke_KG.txt","w")
for index, row in tqdm(df_database.iterrows(), total=len(df_database)):
    if not isinstance(row['relation'], str) or not isinstance(row['Start_entity'], str):
        continue
    start_entity = row["Start_entity"]
    if isinstance(row['Tail_entity2'], str):
        tail_entity = row['Tail_entity2']
    elif not isinstance(row['Tail_entity1'], str) or "http" in row['Tail_entity1']:
        continue
    else:
        tail_entity = row['Tail_entity1']
    relation = row['relation']
    f.write(start_entity)
    f.write("\t")
    f.write(relation)
    f.write("\t")
    f.write(tail_entity)
    f.write("\n")
    f.flush()
f.close()


# tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/Baichuan-7B", trust_remote_code=True)
# model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/Baichuan-7B", trust_remote_code=True).half().cuda()
# model = model.eval()

# df = pd.read_excel("/home/myjia/Medical_LLM_task/dataset/pifuke.xlsx")
# correct_num = 0
# total_num = 0
# for index, row in tqdm(df.iterrows(), total=len(df)):
#     EMR = row["病例详情"]
#     label = row["病例主诊断"]
#     prompt = "以下是一份电子病历,上面记录了病人的现病史、过往史等信息. 请你根据该电子病历，给出你预测的诊断结果。请注意，你给出的预测应该是一个top5列表，列出5个你认为最可能的预测疾病结果，并排名。\n"
#     prompt+="### 电子病历内容:\n{}\n\n### 请给出诊断结果:\n".format(EMR)
#     if len(prompt) > 2048:
#         prompt = prompt[:2048]
#     response, history = model.chat(tokenizer, prompt, history=[])
#     if label in response:
#         correct_num += 1
#     total_num += 1

# print("Accuracy: {}".format(correct_num/total_num))



