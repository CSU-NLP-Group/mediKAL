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

fin = open("/home/myjia/Medical_LLM_task/dataset/CMCQA/CMCQA.json", 'r', encoding='utf-8')

fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/CMCQA/CMCQA_jingshenke.json", 'w', encoding='utf-8')
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/CMCQA/CMCQA_pifuke.json", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/CMCQA/CMCQA_zhongliuke.json", 'w', encoding='utf-8')

def chat(prompt):
    if len(prompt) > 1500:
        prompt = prompt[:1500]
    response, history = chat_model.chat(tokenizer,
                                   prompt,
                                   do_sample=False,
                                   top_p=0.1,
                                   temperature=0.1)
    return response

def final_answer(dialogue_list):
    dialogue_text = ""
    for dialogue in dialogue_list:
        if dialogue[1] == "p":
            dialogue_text += "患者：" + dialogue[0] + "\n"
        else:
            dialogue_text += "医生：" + dialogue[0] + "\n"
    if len(dialogue_text) > 1500:
        dialogue_text = dialogue_text[:1500]
    prompt = "你是一名优秀的AI医生,你可以根据输入的医生与患者的多轮对话，判断该多轮对话属于哪个科室。\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"对话内容：\n\n{dialogue_text}" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

# fin是一个大的list，每个元素是一个dict
data = json.load(fin)
for i, line in tqdm(enumerate(data), total=len(data)):
    # line是一个dict，包含了query，response，label
    dialogue_list = line
    result = final_answer(dialogue_list)
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