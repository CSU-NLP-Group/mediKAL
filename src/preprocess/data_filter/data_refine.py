import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
fin = open("/home/myjia/Medical_LLM_task/LLMs_hub/LLaMA-Factory/data/shuffled_data_train.json", "r", encoding="utf-8")
fout = open("/home/myjia/Medical_LLM_task/MindMap/preprocess/data_filter/train.json", "w", encoding="utf-8")

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
    prompt = "你是一名优秀的AI医生,你可以根据患者所提供的问题，判断该患者应该选择哪个科室。\n\n" \
            + "你只需要回答科室名称即可\n\n" \
            + f"患者描述：{question},\n\n" \
            + "科室名称："
           

    result = chat(prompt=prompt)
    
    return result.strip()

for i, line in tqdm(enumerate(fin)):
    data = json.loads(line)
    conversations = data["conversations"]
    new_conv = {}
    pair_flag = 0
    all_history = []
    history = []
    for j,conv in enumerate(conversations):
        new_conv['input'] = ""
        content = conv["content"]
        if pair_flag == 0:
            new_conv["instruction"] = content
            history.append(content)
            pair_flag = 1
        else:
            new_conv["output"] = content
            history.append(content)
            all_history.append(history)
            if j == 1:
                new_conv['history'] = []
            else:
                new_conv["history"] = all_history
            pair_flag = 0
            total_data.append(new_conv)
            new_conv = {}
            history = []
# 