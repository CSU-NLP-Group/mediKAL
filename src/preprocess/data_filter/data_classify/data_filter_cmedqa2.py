from tqdm import tqdm
import csv
import json
import os
from transformers import AutoTokenizer, AutoModel
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tokenizer = AutoTokenizer.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/home/myjia/Medical_LLM_task/LLMs_base_model/chatglm3-6b", trust_remote_code=True).cuda(0)
chat_model.eval()
fquestion = open("/home/myjia/Medical_LLM_task/dataset/cMedQA2/question.csv", 'r', encoding='utf-8')
fanswer = open("/home/myjia/Medical_LLM_task/dataset/cMedQA2/answer.csv", 'r', encoding='utf-8')
fout_pifuke = open("/home/myjia/Medical_LLM_task/dataset/cMedQA2/cMedQA2_pifuke.json", 'w', encoding='utf-8')
fout_jingshenke = open("/home/myjia/Medical_LLM_task/dataset/cMedQA2/cMedQA2_jingshenke.json", 'w', encoding='utf-8')
fout_zhongliuke = open("/home/myjia/Medical_LLM_task/dataset/cMedQA2/cMedQA2_zhongliuke.json", 'w', encoding='utf-8')


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


# fquestion这个csv文件有两个字段，分别是question_id和content
# fanswer这个csv文件有三个字段，分别是answer_id，question_id和content
# 首先读取fqeustion文件

csv_reader = csv.reader(fquestion)
df = pd.read_csv("/home/myjia/Medical_LLM_task/dataset/cMedQA2/answer.csv")
next(csv_reader)  # 跳过标题行
# 使用csv_reader按行读取，同时用tqdm显示进度条
for row in tqdm(csv_reader):
    question_id = row[0]
    content = row[1]
    result = final_answer(content)
    print("预测结果：{}\n".format(result))
    if "皮肤科" in result:
        # 根据question_id在fanswer中找到对应的行
        question_id_number = int(question_id)
        answer_contents_list = list(set((df.loc[df['question_id']==question_id_number,:]['content'])))
        answer = " ".join(answer_contents_list)
        # 将question和answer的内容写入json文件
        line = {"question": content, "answer": answer}
        fout_pifuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_pifuke.flush()
    elif "精神科" in result:
        question_id_number = int(question_id)
        answer_contents_list = list(set((df.loc[df['question_id']==question_id_number,:]['content'])))
        answer = " ".join(answer_contents_list)
        # 将question和answer的内容写入json文件
        line = {"question": content, "answer": answer}
        fout_jingshenke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_jingshenke.flush()
    elif "肿瘤科" in result:
        question_id_number = int(question_id)
        answer_contents_list = list(set((df.loc[df['question_id']==question_id_number,:]['content'])))
        answer = " ".join(answer_contents_list)
        line = {"question": content, "answer": answer}
        fout_zhongliuke.write(json.dumps(line, ensure_ascii=False) + '\n')
        fout_zhongliuke.flush()

    

