from tqdm import tqdm
import csv
import json
import os
from transformers import AutoTokenizer, AutoModel
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
chat_model.eval()

f1 = open("/home/myjia/Medical_LLM_task/dataset/PromptCBLUE/初赛训练集验证集/train.json", 'r', encoding='utf-8')
f2 = open("/home/myjia/Medical_LLM_task/dataset/PromptCBLUE/初赛训练集验证集/dev.json", 'r', encoding='utf-8')

fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/PromptCBLUE/初赛训练集验证集/PromptCBLUE_jingshenke.json", 'w', encoding='utf-8')
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/PromptCBLUE/初赛训练集验证集/PromptCBLUE_pifuke.json", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/PromptCBLUE/初赛训练集验证集/PromptCBLUE_zhongliuke.json", 'w', encoding='utf-8')

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
    prompt = "你是一名优秀的AI医生,你可以根据输入的文本内容，判断这段文本对应哪个科室。\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"文本输入：{question},\n\n" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

# 按行读取json文件
for i, line in tqdm(enumerate(f1)):
    line = json.loads(line)
    input = line['input']
    result = final_answer(input)
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

for i, line in tqdm(enumerate(f2)):
    line = json.loads(line)
    input = line['input']
    result = final_answer(input)
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