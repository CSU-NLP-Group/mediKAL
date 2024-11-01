from tqdm import tqdm
import csv
import json
import os
from transformers import AutoTokenizer, AutoModel
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
chat_model.eval()

f_train = open("/home/myjia/Medical_LLM_task/dataset/MedQA/data_clean/questions/Mainland/train.jsonl", 'r', encoding='utf-8')
f_dev = open("/home/myjia/Medical_LLM_task/dataset/MedQA/data_clean/questions/Mainland/dev.jsonl", 'r', encoding='utf-8')
f_test = open("/home/myjia/Medical_LLM_task/dataset/MedQA/data_clean/questions/Mainland/test.jsonl", 'r', encoding='utf-8')
f_qbank = open("/home/myjia/Medical_LLM_task/dataset/MedQA/data_clean/questions/Mainland/chinese_qbank.jsonl", 'r', encoding='utf-8')

fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/MedQA/data_clean/questions/Mainland/Mainland_jingshenke.json", 'w', encoding='utf-8')
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/MedQA/data_clean/questions/Mainland/Mainland_pifuke.json", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/MedQA/data_clean/questions/Mainland/Mainland_zhongliuke.json", 'w', encoding='utf-8')

def chat(prompt):
    if len(prompt) > 1500:
        prompt = prompt[:1500]
    response, history = chat_model.chat(tokenizer,
                                   prompt,
                                   do_sample=False,
                                   top_p=0.1,
                                   temperature=0.1)
    return response

def final_answer(question):
    prompt = "你是一名优秀的AI医生,你可以根据所提供的问题，判断该问题属于哪个科室的内容。\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"问题：{question},\n\n" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

# 按行读取json文件
for i, line in tqdm(enumerate(f_train)):
    line = json.loads(line)
    question = line['question']
    options = line['options']
    answer = line['answer']
    meta_info = line['meta_info']
    answer_idx = line['answer_idx']
    result = final_answer(question)
    print("预测结果：{}\n".format(result))
    if "皮肤科" in result:
        fout_pifuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_pifuke.flush()
    elif "精神科" in result:
        fout_jingshenke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_jingshenke.flush()
    elif "肿瘤科" in result:
        fout_zhongliuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_zhongliuke.flush()

for i, line in tqdm(enumerate(f_dev)):
    line = json.loads(line)
    question = line['question']
    options = line['options']
    answer = line['answer']
    meta_info = line['meta_info']
    answer_idx = line['answer_idx']
    result = final_answer(question)
    print("预测结果：{}\n".format(result))
    if "皮肤科" in result:
        fout_pifuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_pifuke.flush()
    elif "精神科" in result:
        fout_jingshenke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_jingshenke.flush()
    elif "肿瘤科" in result:
        fout_zhongliuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_zhongliuke.flush()

for i, line in tqdm(enumerate(f_test)):
    line = json.loads(line)
    question = line['question']
    options = line['options']
    answer = line['answer']
    meta_info = line['meta_info']
    answer_idx = line['answer_idx']
    result = final_answer(question)
    print("预测结果：{}\n".format(result))
    if "皮肤科" in result:
        fout_pifuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_pifuke.flush()
    elif "精神科" in result:
        fout_jingshenke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_jingshenke.flush()
    elif "肿瘤科" in result:
        fout_zhongliuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_zhongliuke.flush()

for i, line in tqdm(enumerate(f_qbank)):
    line = json.loads(line)
    question = line['question']
    options = line['options']
    answer = line['answer']
    # meta_info = line['meta_info']
    # answer_idx = line['answer_idx']
    result = final_answer(question)
    print("预测结果：{}\n".format(result))
    if "皮肤科" in result:
        fout_pifuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_pifuke.flush()
    elif "精神科" in result:
        fout_jingshenke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_jingshenke.flush()
    elif "肿瘤科" in result:
        fout_zhongliuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_zhongliuke.flush()