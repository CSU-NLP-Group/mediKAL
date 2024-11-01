from tqdm import tqdm
import csv
import json
import os
from transformers import AutoTokenizer, AutoModel
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
chat_model.eval()

fin = open("/home/myjia/Medical_LLM_task/dataset/qizhen/qizhen-sft-20k.json", 'r', encoding='utf-8')

fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/qizhen/qizhen_jingshenke.json", 'w', encoding='utf-8')
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/qizhen/qizhen_pifuke.json", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/qizhen/qizhen_zhongliuke.json", 'w', encoding='utf-8')

def chat(prompt):
    if len(prompt) > 1500:
        prompt = prompt[:1500]
    response, history = chat_model.chat(tokenizer,
                                   prompt,
                                   do_sample=False,
                                   top_p=0.1,
                                   temperature=0.1)
    return response

def final_answer(question, answer):
    prompt = "你是一名优秀的AI医生,你可以根据输入的医疗问答对，判断该问答对属于哪个科室相关的问题。\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"问题：{question},\n\n 回答：{answer}" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

# fin是一个大的list，每个元素是一个dict
data = json.load(fin)
for i, line in tqdm(enumerate(data), total=len(data)):
    # line是一个dict，包含了query，response，label
    instruction = line['instruction']
    output = line['output']
    result = final_answer(instruction, output)
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