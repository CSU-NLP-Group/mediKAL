# 这个实验还没跑
#################################################################
from tqdm import tqdm
import csv
import json
import os
from transformers import AutoTokenizer, AutoModel
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
chat_model.eval()


fin = open("/home/myjia/Medical_LLM_task/dataset/DoctorGLM/DoctorGLM_disease_info.json", 'r', encoding='utf-8')
fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/DoctorGLM/DoctorGLM_jingshenke.json", 'w', encoding='utf-8')
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/DoctorGLM/DoctorGLM_pifuke.json", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/DoctorGLM/DoctorGLM_zhongliuke.json", 'w', encoding='utf-8')

def chat(prompt):
    if len(prompt) > 1500:
        prompt = prompt[:1500]
    response, history = chat_model.chat(tokenizer,
                                   prompt,
                                   do_sample=False,
                                   top_p=0.1,
                                   temperature=0.1)
    return response

def final_answer(disease):
    prompt = "你是一名优秀的AI医生,你可以根据输入的疾病名称，判断该疾病属于哪个科室。\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"疾病名称：{disease},\n\n" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

# fin是一个大的dict，每个元素是一个dict
data = json.load(fin)
for i,key in tqdm(enumerate(data.keys()), total=len(data.keys())):
    result = final_answer(key)
    print("预测结果：{}\n".format(result))
    if "皮肤科" in result:
        cur_dict = data[key]
        cur_dict['疾病名称'] = key
        fout_pifuke.write(json.dumps(data[key], ensure_ascii=False) + '\n')
        fout_pifuke.flush()
    elif "精神科" in result:
        cur_dict = data[key]
        cur_dict['疾病名称'] = key
        fout_jingshenke.write(json.dumps(data[key], ensure_ascii=False) + '\n')
        fout_jingshenke.flush()
    elif "肿瘤科" in result:
        cur_dict = data[key]
        cur_dict['疾病名称'] = key
        fout_zhongliuke.write(json.dumps(data[key], ensure_ascii=False) + '\n')
        fout_zhongliuke.flush()

